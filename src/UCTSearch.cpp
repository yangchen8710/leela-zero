/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/





#include "config.h"
#include "UCTSearch.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <fstream>
#include <iostream>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"

using namespace Utils;

#define M_LOG2E 1.44269504088896340736 // log2(e)

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

UCTSearch::UCTSearch(GameState& g)
    : m_rootstate(g) {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
}

bool UCTSearch::advance_to_new_rootstate() {
    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth =
        int(m_rootstate.get_movenum() - m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }


    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // Make sure that the nodes we destroyed the previous move are
    // in fact destroyed.
    while (!m_delete_futures.empty()) {
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        ThreadGroup tg(thread_pool);

        test->forward_move();
        const auto move = test->get_last_move();

        auto oldroot = std::move(m_root);
        m_root = oldroot->find_child(move);

        // Lazy tree destruction.  Instead of calling the destructor of the
        // old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread.  This will save a
        // bit of time when dealing with large trees.
        auto p = oldroot.release();
        tg.add_task([p]() { delete p; });
        m_delete_futures.push_back(std::move(tg));

        if (!m_root) {
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::update_root() {
    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    m_playouts = 0;

#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes();
#endif

    //if (!advance_to_new_rootstate() || !m_root) {
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    //}
    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes();

#ifndef NDEBUG
    if (m_nodes > 0) {
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }
#endif
}

float UCTSearch::get_min_psa_ratio() const {
    const auto mem_full = m_nodes / static_cast<float>(MAX_TREE_SIZE);
    // If we are halfway through our memory budget, start trimming
    // moves with very low policy priors.
    if (mem_full > 0.5f) {
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) {
            return 0.01f;
        }
        return 0.001f;
    }
    return 0.0f;
}

SearchResult UCTSearch::play_simulation_sh(GameState & currstate,
	UCTNode* const node) {
	const auto color = currstate.get_to_move();
	auto result = SearchResult{};

	node->virtual_loss();

	if (node->expandable()) {
		if (currstate.get_passes() >= 2) {
			auto score = currstate.final_score();
			result = SearchResult::from_score(score);
		}
		else if (m_nodes < MAX_TREE_SIZE) {
			float eval;
			const auto had_children = node->has_children();
			const auto success =
				node->create_children(m_nodes, currstate, eval,
					get_min_psa_ratio());
			if (!had_children && success) {
				result = SearchResult::from_eval(eval);
			}
		}
	}

	if (node->has_children() && !result.valid()) {
		auto next = node->uct_select_child(color, node == m_root.get());
		auto move = next->get_move();

		currstate.play_move(move);
		if (move != FastBoard::PASS && currstate.superko()) {
			next->invalidate();
		}
		else {
			result = play_simulation_sh(currstate, next);
		}
	}

	if (result.valid()) {
		node->update(result.eval());
	}
	node->virtual_loss_undo();

	return result;
}

SearchResult UCTSearch::play_simulation(GameState & currstate,
                                        UCTNode* const node) {
    const auto color = currstate.get_to_move();
    auto result = SearchResult{};

    node->virtual_loss();

    if (node->expandable()) {
        if (currstate.get_passes() >= 2) {
            auto score = currstate.final_score();
            result = SearchResult::from_score(score);
        } else if (m_nodes < MAX_TREE_SIZE) {
            float eval;
            const auto had_children = node->has_children();
            const auto success =
                node->create_children(m_nodes, currstate, eval,
                                      get_min_psa_ratio());
            if (!had_children && success) {
                result = SearchResult::from_eval(eval);
            }
        }
    }

    if (node->has_children() && !result.valid()) {
        auto next = node->uct_select_child(color, node == m_root.get());
        auto move = next->get_move();

        currstate.play_move(move);
        if (move != FastBoard::PASS && currstate.superko()) {
            next->invalidate();
        } else {
            result = play_simulation(currstate, next);
        }
    }

    if (result.valid()) {
        node->update(result.eval());
    }
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    // sort children, put best move on top
    parent.sort_children(color);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        std::string move = state.move_to_text(node->get_move());
        FastState tmpstate = state;
        tmpstate.play_move(node->get_move());
        std::string pv = move + " " + get_pv(tmpstate, *node);

        myprintf("%4s -> %7d (V: %5.2f%%) (N: %5.2f%%) PV: %s\n",
            move.c_str(),
            node->get_visits(),
            node->get_visits() ? node->get_eval(color)*100.0f : 0.0f,
            node->get_score() * 100.0f,
            pv.c_str());
    }
    tree_stats(parent);
}

void tree_stats_helper(const UCTNode& node, size_t depth,
                       size_t& nodes, size_t& non_leaf_nodes,
                       size_t& depth_sum, size_t& max_depth,
                       size_t& children_count) {
    nodes += 1;
    non_leaf_nodes += node.get_visits() > 1;
    depth_sum += depth;
    if (depth > max_depth) max_depth = depth;

    for (const auto& child : node.get_children()) {
        if (child.get_visits() > 0) {
            children_count += 1;
            tree_stats_helper(*(child.get()), depth+1,
                              nodes, non_leaf_nodes, depth_sum,
                              max_depth, children_count);
        } else {
            nodes += 1;
            depth_sum += depth+1;
            if (depth+1 > max_depth) max_depth = depth+1;
        }
    }
}

void UCTSearch::tree_stats(const UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;
    tree_stats_helper(node, 0,
                      nodes, non_leaf_nodes, depth_sum,
                      max_depth, children_count);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n",
                 non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

bool UCTSearch::should_resign(passflag_t passflag, float bestscore) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t board_squares = m_rootstate.board.get_boardsize()
                               * m_rootstate.board.get_boardsize();
    const auto move_threshold = board_squares / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (bestscore > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * board_squares));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (bestscore > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    // Make sure best is first
    m_root->sort_children(color);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    auto bestscore = first_child->get_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    bestscore = 1.0f;
                } else {
                    bestscore = nopass->get_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else {
        if (!cfg_dumbpass && bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.
            float score = m_rootstate.final_score();
            // Do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        bestscore = 1.0f;
                    } else {
                        bestscore = nopass->get_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else {
                myprintf("Passing wins :-)\n");
            }
        } else if (!cfg_dumbpass
                   && m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?
            float score = m_rootstate.final_score();
            // do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses, I'll play on.\n");
            } else {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, bestscore)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * bestscore);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

void UCTSearch::dump_analysis(int playouts) {
    if (cfg_quiet) {
        return;
    }

    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    std::string pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_eval(color);
    myprintf("Playouts: %d, Win: %5.2f%%, PV: %s\n",
             playouts, winrate, pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && m_nodes < MAX_TREE_SIZE;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move) const {
    auto playouts = m_playouts.load();
    const auto playouts_left =
        std::max(0, std::min(m_maxplayouts - playouts,
                             m_maxvisits - m_root->get_visits()));

    // Wait for at least 1 second and 100 playouts
    // so we get a reliable playout_rate.
    if (elapsed_centis < 100 || playouts < 100) {
        return playouts_left;
    }
    const auto playout_rate = 1.0f * playouts / elapsed_centis;
    const auto time_left = std::max(0, time_for_move - elapsed_centis);
    return std::min(playouts_left,
                    static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_noncontenders(int elapsed_centis, int time_for_move) {
    auto Nfirst = 0;
    // There are no cases where the root's children vector gets modified
    // during a multithreaded search, so it is safe to walk it here without
    // taking the (root) node lock.
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            Nfirst = std::max(Nfirst, node->get_visits());
        }
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left(elapsed_centis, time_for_move);
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto has_enough_visits =
                node->get_visits() >= min_required_visits;

            node->set_active(has_enough_visits);
            if (!has_enough_visits) {
                ++pruned_nodes;
            }
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(int elapsed_centis, int time_for_move) {
    if (cfg_timemanage == TimeManagement::OFF) {
        return true;
    }
    auto pruned = prune_noncontenders(elapsed_centis, time_for_move);
    if (pruned < m_root->get_children().size() - 1) {
        return true;
    }
    // If we cannot save up time anyway, use all of it. This
    // behavior can be overruled by setting "fast" time management,
    // which will cause Leela to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot
    // think ahead about her next moves in the remaining time.
    auto my_color = m_rootstate.get_to_move();
    auto tc = m_rootstate.get_timecontrol();
    if (!tc.can_accumulate_time(my_color)
        || m_maxplayouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_timemanage != TimeManagement::FAST) {
            return true;
        }
    }
    // In a timed search we will essentially always exit because
    // the remaining time is too short to let another move win, so
    // avoid spamming this message every move. We'll print it if we
    // save at least half a second.
    if (time_for_move - elapsed_centis > 50) {
        myprintf("%.1fs left, stopping early.\n",
                    (time_for_move - elapsed_centis) / 100.0f);
    }
    return false;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move) const {
    return m_playouts >= m_maxplayouts
           || m_root->get_visits() >= m_maxvisits
           || elapsed_centis >= time_for_move;
}

void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = m_search->play_simulation(*currstate, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::gen_policy_move(GameState& state, Random rd)
{
	const auto raw_netlist = Network::get_scored_moves(
		&state, Network::Ensemble::RANDOM_SYMMETRY);
	if (raw_netlist.winrate > 0.9 || raw_netlist.winrate < 0.1)
		return FastBoard::PASS;

	std::vector<Network::ScoreVertexPair> nodelist;
	auto to_move = state.board.get_to_move();

	auto legal_sum = 0.0f;
	for (auto i = 0; i < BOARD_SQUARES; i++) {
		const auto x = i % BOARD_SIZE;
		const auto y = i / BOARD_SIZE;
		const auto vertex = state.board.get_vertex(x, y);
		if (state.is_move_legal(to_move, vertex)) {
			nodelist.emplace_back(raw_netlist.policy[i], vertex);
			legal_sum += raw_netlist.policy[i];
		}
	}
	nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
	legal_sum += raw_netlist.policy_pass;

	if (legal_sum > std::numeric_limits<float>::min()) {
		// re-normalize after removing illegal moves.
		for (auto& node : nodelist) {
			node.first /= legal_sum;
		}
	}
	else {
		// This can happen with new randomized nets.
		auto uniform_prob = 1.0f / nodelist.size();
		for (auto& node : nodelist) {
			node.first = uniform_prob;
		}
	}

	int max = 100000;
	double prop_sum = 0.0;
	//auto move_idx = rd.randuint64(legal_moves.size());
	auto move_rd = 1.0 * (rand() % max) / max;

	for (int tmpj = 0; tmpj < nodelist.size(); tmpj++)
	{
		prop_sum += nodelist[tmpj].first;
		if(move_rd <= prop_sum)
			return nodelist[tmpj].second;
	}
	return nodelist[nodelist.size()-1].second;
}


int UCTSearch::gen_random_move(GameState& state, Random rd)
{
	auto to_move = state.board.get_to_move();
	std::vector<int> legal_moves;
	for (auto i = 0; i < BOARD_SQUARES; i++) {
		const auto x = i % BOARD_SIZE;
		const auto y = i / BOARD_SIZE;
		const auto vertex = state.board.get_vertex(x, y);
		if (state.is_move_legal(to_move, vertex)) {
			legal_moves.emplace_back(vertex);
		}
	}
	if (legal_moves.size() <= 20)
		return -1;


	//auto move_idx = rd.randuint64(legal_moves.size());
	auto move_idx = rand() % legal_moves.size();

	return legal_moves[move_idx];
}

int UCTSearch::random_playout(GameState& state, Random rd, int mode)
{

	int side = state.get_to_move();
	float winrate;
	int res_move_old, res_move_new = 1;
	int gen_moves = 0;
	//Time start;
	do
	{
		res_move_old = res_move_new;
		switch (mode)
		{
		case 0:
			res_move_new = gen_random_move(state, rd);
		case 1:
			res_move_new = gen_policy_move(state, rd);
		default:
			break;
		}
		auto to_move = state.board.get_to_move();
		state.play_move(res_move_new);
		gen_moves++;
		//const auto raw_netlist = Network::get_scored_moves(
		//	&state, Network::Ensemble::RANDOM_SYMMETRY);
		//if (raw_netlist.winrate > 0.95 || raw_netlist.winrate < 0.05)
		//	break;


	} while (res_move_new != FastBoard::PASS && res_move_old != FastBoard::PASS);
	const auto raw_netlist = Network::get_scored_moves(
		&state, Network::Ensemble::RANDOM_SYMMETRY);

	// DCNN returns winrate as side to move
	if (state.get_to_move() == side)
		winrate = raw_netlist.winrate;
	else
		winrate = 1.0f - raw_netlist.winrate;
	//Time elapsed;
	//int elapsed_centis = Time::timediff_centis(start, elapsed);
	//myprintf("side = %d,winrate = %f,gen_moves=%d ,elapsed_centis=%d\n",
	//	side,winrate, gen_moves, elapsed_centis);
	//state.display_state();
	if (winrate > 0.5)
		return 1;
	else
		return -1;
}

std::vector<int> UCTSearch::sort_round_children(std::vector<int> child_in_round, UCTNode* node)
{
	std::vector<float> child_sh_score_in_round;

	int n = child_in_round.size();
	for (int tmpj = 0; tmpj < n; tmpj++)
	{
		int child_idx = child_in_round[tmpj];
		double child_rp_count = 1.0* node->m_children[child_idx]->shot_po_count;
		double child_rp_win = node->m_children[tmpj]->shot_wins;
		double child_rp_winrate;
		if (child_rp_win == 0)
			child_rp_winrate = 0;
		else
			child_rp_winrate = 1.0f * child_rp_win / child_rp_count;
		child_sh_score_in_round.emplace_back(child_rp_winrate);
	}

	for (int i = 0; i < n- 1; i++) {
		for (int j = 0; j < n - i - 1; j++) {
			if (child_sh_score_in_round[j] < child_sh_score_in_round[j + 1]) {
				float tempf = child_sh_score_in_round[j];
				int tempi = child_in_round[j];
				child_sh_score_in_round[j] = child_sh_score_in_round[j + 1];
				child_in_round[j] = child_in_round[j + 1];
				child_sh_score_in_round[j + 1] = tempf;
				child_in_round[j + 1] = tempi;
			}
		}
	}
	return child_in_round;

}


std::vector<int> UCTSearch::get_new_round_children(std::vector<int> child_in_round,UCTNode* node)
{
	std::vector<int> new_children_in_round;
	std::vector<float> child_sh_score_in_round;

	//child_in_round = sort_round_children(child_in_round, node);

	int n = child_in_round.size();
	n = ceil(1.0 * n / 2);

	for (int tmpj = 0; tmpj < n; tmpj++)
	{
		new_children_in_round.emplace_back(child_in_round[tmpj]);
	}

	return new_children_in_round;
}

int UCTSearch::think_sh(int color, passflag_t passflag,int bestmove) {
	
	int coin = 50000;
	update_root();
	Random rd = Random(time(NULL));
	srand((unsigned)time(NULL));
	// set side to move
	m_rootstate.board.set_to_move(color);

	

	m_root->prepare_root_node(color, m_nodes, m_rootstate);
	
	int child_count = m_root->m_children.size()-1;
	int round_count = log(child_count) * M_LOG2E;
	int round_coin = coin / round_count;

	myprintf("child_count = %d, round_count = %d, round_coin = %d,  \n",
		child_count,
		round_count,
		round_coin);

	//TODO:children's eval
	for (int tmpj = 0; tmpj < child_count + 1; tmpj++)
	{
		auto node = m_root->m_children[tmpj].get();
		auto currstate = std::make_unique<GameState>(m_rootstate);
		currstate->play_move(m_root->m_children[tmpj]->get_move());
		float root_eval;
		if (node->expandable())
		{
			node->create_children(m_nodes, *currstate, root_eval);
		}
	}


	//rp_count[tmpj], rp_win[tmpj]);


	std::vector<int> child_in_round;

	for (int tmpj = 0; tmpj < child_count + 1; tmpj++)
	{
		if(m_root->m_children[tmpj].get_move()!=FastBoard::PASS)
			child_in_round.emplace_back(tmpj);
	}

	round_count = 1;
	while (child_in_round.size() != 1)
	{
		int old_r = round_count;
		int node_coin = round_coin / child_in_round.size();
		if (round_count == 1)
		{
			myprintf("child_in_round.size = %d, node_coin = %d, round_coin = %d,  \n",
				child_in_round.size(),
				node_coin,
				round_coin);
		}
		
		for (int tmpj = 0; tmpj < child_in_round.size(); tmpj++)
		{
			int child_idx = child_in_round[tmpj];
			for (int tmpi = 0; tmpi < node_coin; tmpi++)
			{
				auto currstate = std::make_unique<GameState>(m_rootstate);
				currstate->play_move(m_root->m_children[child_idx]->get_move());
				auto rp_res = -random_playout(*currstate, rd);
				//rp_count[tmpj] += 1;
				m_root->m_children[child_idx]->add_random_playouts_count();
				if (rp_res == 1)
					m_root->m_children[child_idx]->add_random_playouts_win();
				//rp_win[tmpj] += 1;
			}
		}

		child_in_round = get_new_round_children(child_in_round,m_root.get());
		for (int tmpi = 0; tmpi < child_in_round.size(); tmpi++)
		{
			if (m_root->m_children[child_in_round[tmpi]]->get_move() == bestmove)
			{
				myprintf("best move in round%d.\n", round_count);
				round_count++;
				break;
			}
		}
		if (old_r == round_count)
			return 0;
	}

	myprintf("move\tpolicy\t\teval\trp_count\trp_wins\n");
	for (int tmpj = 0; tmpj < m_root->m_children.size(); tmpj++)
	{
		std::string vertex = m_rootstate.move_to_text(m_root->m_children[tmpj].get_move());

		myprintf("%s\t%f\t%f\t%d\t%d\n",
			vertex.c_str(),
			m_root->m_children[tmpj]->m_score,
			m_root->m_children[tmpj]->m_net_eval,
			m_root->m_children[tmpj]->random_playouts_count, 
			m_root->m_children[tmpj]->random_playouts_win);
			//rp_count[tmpj], rp_win[tmpj]);
	}
	return m_root->m_children[child_in_round[0]]->get_move();

}

std::vector<int>  UCTSearch::get_legal_moves(GameState& currstate)
{
	std::vector<int> legal_moves;
	auto to_move = currstate.board.get_to_move();
	for (auto i = 0; i < BOARD_SQUARES; i++) {
		const auto x = i % BOARD_SIZE;
		const auto y = i / BOARD_SIZE;
		const auto vertex = currstate.board.get_vertex(x, y);
		if (currstate.is_move_legal(to_move, vertex)) {
			legal_moves.emplace_back(vertex);
		}
	}
	return legal_moves;
}
int checkwin(GameState& state)
{
	const auto raw_netlist = Network::get_scored_moves(
		&state, Network::Ensemble::RANDOM_SYMMETRY);

	if (raw_netlist.winrate > 0.5)
		return 1;
	else
		return -1;
}
int progressivew(int n)
{
	double t = 0;
	for (int x = 1;; x++)
	{
		double bei = 1.0;
		for (int y = 0; y < x - 1; y++)
			bei = bei * 3.14;
		t = t + 40 * bei;
		if (n < t)
			return x + 1;
	}
}

double UCTSearch::shot(GameState& currstate, 
	UCTNode* node, Random& rd, 
	int buget,int& budgetUsed,int& playouts,
	double& wins,bool isroot,
	int po_res_mode ,int pw,
	int bestmove,int mixmax,
	int width)
{	//myprintf("isroot %d.\n", isroot);
	//myprintf("buget %d, budgetUsed %d, playouts %d, wins %d, \n", buget, budgetUsed, playouts, wins);
		
	//thesis algorithm:board is terminal
	if (node->m_children.size() == 1)//terminal
	{
		double win;
		double evelf;
		evelf = node->get_eval(currstate.get_to_move());
		switch (po_res_mode)
		{
			case 0:
			case 1:
				if (evelf > 0.5)
					win = 1;
				else
					win = 0;
				win = 1 - win;
			case 2:
				win = evelf;
		}

		node->update_shot(buget, win * buget);
		playouts += buget;
		wins += 1.0* playouts -win * buget;
		return win;
	}
	if (buget == 0)
		return 0;
	//thesis algorithm:buget == 1
	if (buget == 1)
	{
		double result;
		int res_int;
		auto resrstate = std::make_unique<GameState>(currstate);
		switch (po_res_mode)
		{
			case 0://use random playout
				res_int = random_playout(*resrstate, rd,0);
				//myprintf("isroot %d.\n", res_int);
				//while loss, random_playout returns -1
				if (res_int == 1)
					result = 1.0;
				else
					result = 0.0;
				result = 1.0 - result;
				break;
			case 1://use policy playout
				res_int = random_playout(*resrstate, rd,1);
				//while loss, random_playout returns -1
				if (res_int == 1)
					result = 1.0;
				else
					result = 0.0;
				result = 1.0 - result;
				break;
			case 2://use value net instead of random playout
				const auto raw_netlist = Network::get_scored_moves(
					&currstate, Network::Ensemble::RANDOM_SYMMETRY);
				result = 1- raw_netlist.winrate;
				
				break;
		}
		//myprintf("win %f.\n", result);
		node->update_shot(1, result);
		wins += 1-result;
		budgetUsed++;
		playouts++;
		return  1-result;
		
	}
	//create chilren
	float eval;
	if (!node->has_children())
	{
		node->create_children(m_nodes, currstate, eval);
		node->inflate_all_children();
	}
	
	//thesis algorithm:S<-possible moves
	//for now, only pick 16 moves with best policy network
	std::vector<int> child_in_round;
	int bugetSum = node->shot_po_count + buget;
	int nsize;
	int added = 0;
	if (width < 0)
		nsize = node->m_children.size() - 1;
	else
		nsize = width;
	if (pw == 1)
	{
		nsize = progressivew(bugetSum);
	}

	//myprintf("pwmode=%d,nsize %d\n", pw,nsize);

	for (int tmpj = 0; added < nsize; tmpj++)
	{
		//myprintf("%f\n",node->m_children[tmpj].get_score());
		if (tmpj >= node->m_children.size())
			break;
		if (node->m_children[tmpj].get_move() != FastBoard::PASS)
		{
			child_in_round.emplace_back(tmpj);
			added++;
			//myprintf("(*nodexx).get_move %d\n", node->m_children[tmpj].get_move());
		}
			
	}
	//myprintf("child_in_round.size() %d,node->m_children %d\n", child_in_round.size(), node->m_children.size());
	//myprintf("\n");
	//thesis algorithm:if|S|==1
	if (child_in_round.size() == 1)
	{
		int nu, np;
		double nw;
		nu = np = nw = 0;
		auto nextstate = std::make_unique<GameState>(currstate);
		nextstate->play_move(node->m_children[child_in_round[0]]->get_move());
		
		shot(*nextstate, (node->m_children[child_in_round[0]]).get(), rd, buget, nu, np, nw,false, po_res_mode, pw,bestmove,mixmax,width);
		budgetUsed += nu;
		playouts += np;
		wins += 1.0*np - nw;
		//update
		node->update_shot(np, nw);
	}
	
	//thesis algorithm:if t.budgetNode <= |S|
	//do playout at children node at least for one time
	int budgetNode = node->shot_po_count;
	int playedBudget = 0;
	//myprintf("child_in_round.size() %d,node->m_children %d\n", child_in_round.size(), node->m_children.size());

	if (budgetNode <= child_in_round.size())
	{
		int usedinthisfor = 0;
		double winsthisfor = 0;
		for (int tmpi = 0; tmpi < child_in_round.size(); tmpi++)
		{
			node->inflate_all_children();
			int child_idx = child_in_round[tmpi];
			if (budgetUsed >= buget)
			{
				
				
				double reswin;
				if (mixmax)
				{
					//sort
					{
						int tpn = child_in_round.size();
						for (int tmpi = 0; tmpi < tpn - 1; tmpi++) {
							for (int tmpj = 0; tmpj < tpn - tmpi - 1; tmpj++) {
								double child_rp_count = 1.0* node->m_children[child_in_round[tmpj]]->shot_po_count;
								double child_rp_win = node->m_children[child_in_round[tmpj]]->shot_wins;
								double child_rp_counto = 1.0* node->m_children[child_in_round[tmpj + 1]]->shot_po_count;
								double child_rp_wino = node->m_children[child_in_round[tmpj + 1]]->shot_wins;
								if (child_rp_win / child_rp_count < child_rp_wino / child_rp_counto) {
									int tempi = child_in_round[tmpj];
									child_in_round[tmpj] = child_in_round[tmpj + 1];
									child_in_round[tmpj + 1] = tempi;
								}
							}
						}
					}
					double best_rate = node->m_children[child_in_round[0]]->shot_wins / node->m_children[child_in_round[0]]->shot_po_count;
					reswin = (1 - best_rate) * usedinthisfor;
					wins += (best_rate)* usedinthisfor;
				}
				else
				{
					reswin = winsthisfor;
				}
				//update
				node->update_shot(usedinthisfor, reswin);
				return winsthisfor/usedinthisfor;
			}
				
			auto& nodex = node->m_children[child_idx];
			//myprintf("child_idx %d,child_in_round.size() %d,node->m_children %d\n", child_idx, child_in_round.size(), node->m_children.size());
			auto nodexx = nodex.get();
			//myprintf("(*nodexx).get_move %d\n", (*nodexx).get_move());
			if ((*nodexx).shot_po_count == 0)
			{
				auto nextstate = std::make_unique<GameState>(currstate);
				//myprintf("color %d,", nextstate->get_to_move());
				nextstate->play_move(node->m_children[child_idx]->get_move());
				int nu, np;
				double nw;
				nu = np = nw = 0;
				//myprintf("color %d\n", nextstate->get_to_move());
				shot(*nextstate, (node->m_children[child_idx].get()), rd, 1, nu, np, nw,false, po_res_mode, pw,bestmove, mixmax,width);
				usedinthisfor +=nu;
				winsthisfor += nw;
				budgetUsed += nu;
				playouts += np;
				wins += 1.0*np - nw;
				//update
				//node->update_shot(np, nw);
				playedBudget++;
			}
		}
	}
	
	//thesis algorithm:sort moves in S according to their mean
	sort_round_children(child_in_round, node);

	//thesis algorithm:while |S|>1 do
	budgetNode = node->shot_po_count;
	int childSum = child_in_round.size();
	int round_sum = ceil(log(childSum) * M_LOG2E);
	int po_sum_to_increase = 0;
	int round_no = 0;
	int roundbudgetsSumUpdate = 0;

	//myprintf("buget %d,round_sum %d,child_in_round.size() %d\n", buget, round_sum, child_in_round.size());
	int flag = 1;
	while (child_in_round.size() > 1)
	{
		round_no++;
		/*
		if (isroot)
		{
			myprintf("round %d,child_in_round.size() %d,po_sum_to_increase %d\n", round_no, child_in_round.size(), po_sum_to_increase);
		}*/

		
			
		int roundbudgetsSum = (bugetSum) / (round_sum);
		roundbudgetsSumUpdate += roundbudgetsSum;
		int node_budgets = roundbudgetsSum / (child_in_round.size());
		//if ((budgetNode + buget) / round_sum != node_budgets * child_in_round.size())
		int diff = roundbudgetsSum - node_budgets * child_in_round.size();
		//if(isroot)
		//	myprintf("diff is %d\n", diff);
		
		if(diff!=0)
		{
			//if (node_budgets == 0)
			//	flag = 0;
			//if (flag)
			{
				diffsum += diff;
				//myprintf("budget_sum %d,%d nodes,nodes_budgets %d\n", bugetSum / round_sum, child_in_round.size(), node_budgets);
				//myprintf("diff is %d\n", (bugetSum) / round_sum - node_budgets * child_in_round.size());
			}
			
		}
			
			

		node_budgets = 1 > node_budgets ? 1 : node_budgets;
		po_sum_to_increase += node_budgets;
		//po_sum_to_increase = (bugetSum - node->shot_po_count) / (round_sum - round_no+1)/ child_in_round.size();
		//po_sum_to_increase = (round_no * roundbudgetsSum - node->shot_po_count)/ child_in_round.size();
		//myprintf("node_budgets %d,po_sum_to_increase %d\n", node_budgets, po_sum_to_increase);
		

		for (int tmpi = 0; tmpi < child_in_round.size(); tmpi++)
		{
			//myprintf("tmpi %d,child_in_round[tmpi] %d\n", tmpi, child_in_round[tmpi]);
			int po_times; 
			//int d = (*(node->m_children[1])).get_score();
			node->inflate_all_children();
			if (node->m_children[child_in_round[tmpi]]->shot_po_count < po_sum_to_increase)
			{
				po_times = po_sum_to_increase - node->m_children[child_in_round[tmpi]]->shot_po_count;
				if (tmpi < diff)
					po_times++;
				//myprintf("node_budgets %d,po_sum_to_increase %d,po_times %d,node_po %d\n", node_budgets, po_sum_to_increase, po_times, node->m_children[child_in_round[tmpi]]->shot_po_count);
				po_times = po_times < (buget - budgetUsed) ? po_times : (buget - budgetUsed);
				if (po_times == 0)
				{
					continue;
				}
				//myprintf("po_times %d,node_po %d\n", po_times, node->m_children[child_in_round[tmpi]]->shot_po_count);

				//if root
				/*
				if (isroot && child_in_round.size() == 2 && tmpi == 0)
				{
				po_times = buget - budgetUsed;
				po_times = po_times / 2;

				}
				*/
				int nu, np;
				double nw;
				nu = np = nw = 0;
				auto nextstate = std::make_unique<GameState>(currstate);
				nextstate->play_move(node->m_children[child_in_round[tmpi]]->get_move());

				double winrate = shot(*nextstate, (node->m_children[child_in_round[tmpi]]).get(), rd, po_times, nu, np, nw, false, po_res_mode, pw,bestmove,mixmax, width);
				//node->m_children[child_in_round[tmpi]]->update_shot(np,nw);
				//if(po_times!= nu)
				//	myprintf("po_times %d,budgetUsed %d,playouts %d,after_po %d\n", po_times, nu, np, node->m_children[child_in_round[tmpi]]->shot_po_count);
				if (0)
				{
					std::string vertex;
					vertex = m_rootstate.move_to_text(m_root->m_children[child_in_round[tmpi]]->get_move());
					node->m_children[child_in_round[tmpi]]->get_move();
					myprintf("move %s,np %d,nw %f,1.0*np - nw %f\n", vertex.c_str(),np, nw, 1.0*np - nw);
				}
				budgetUsed += nu;
				playouts += np;
				if (!mixmax)
				{
					wins += 1.0*np - nw;
					node->update_shot(np, nw);
				}

				//update
			}
				if (budgetUsed >= buget)
				{
					break;
				}

			
		}
		//if (isroot)
		//{
		//	myprintf("budgetUsed %d,playouts %d,wins %f\n", budgetUsed, playouts, wins);
		//}
		//else if (diff != 0)
		//myprintf("budgetUsed %d,playouts %d,wins %f\n", budgetUsed, playouts, wins);
		
		int successflag = 0;
		node->inflate_all_children();

		{
			int tpn = child_in_round.size();

			for (int tmpi = 0; tmpi < tpn - 1; tmpi++) {
				for (int tmpj = 0; tmpj < tpn - tmpi - 1; tmpj++) {
					double child_rp_count = 1.0* node->m_children[child_in_round[tmpj]]->shot_po_count;
					double child_rp_win = node->m_children[child_in_round[tmpj]]->shot_wins;
					double child_rp_counto = 1.0* node->m_children[child_in_round[tmpj+1]]->shot_po_count;
					double child_rp_wino = node->m_children[child_in_round[tmpj+1]]->shot_wins;
					if (child_rp_win/ child_rp_count < child_rp_wino / child_rp_counto) {
						int tempi = child_in_round[tmpj];
						child_in_round[tmpj] = child_in_round[tmpj + 1];
						child_in_round[tmpj + 1] = tempi;
					}
				}
			}
		}
		
		if (isroot && bestmove > -1)
		{
			for (int tmpi = 0; tmpi < child_in_round.size() && tmpi <10; tmpi++)
			{
				std::string vertex;
				vertex = m_rootstate.move_to_text(m_root->m_children[child_in_round[tmpi]]->get_move());
				myprintf("move:%s,wins %f,po %d,rate %f\n",
					vertex.c_str(),
					m_root->m_children[child_in_round[tmpi]]->shot_wins,
					m_root->m_children[child_in_round[tmpi]]->shot_po_count,
					1.0*m_root->m_children[child_in_round[tmpi]]->shot_wins / m_root->m_children[child_in_round[tmpi]]->shot_po_count);
			}
			myprintf("\n\n\n\n\n\n");
		}
		
		child_in_round = get_new_round_children(child_in_round, node);

		if (isroot && bestmove>-1)
		{
			
			
			for (int tmpi = 0; tmpi < child_in_round.size(); tmpi++)
			{
				if (m_root->m_children[child_in_round[tmpi]]->get_move() == bestmove)
				{
					myprintf("best move in round%d,buget %d, budgetUsed %d, playouts %d, wins %f\n", round_no, buget, budgetUsed, playouts, wins);

					successflag = 1;
					break;
				}
			}
			if (!successflag)
				return 0;
		}

		if(budgetUsed>=buget)
		{
			break;
		}
		//if (isroot)
		//	myprintf("update:buget %d, budgetUsed %d, playouts %d, wins %f, \n", buget, budgetUsed, playouts, wins);
	}
	//thesis algorithm:return first move of S
	if (isroot)
		return node->m_children[child_in_round[0]].get_move();
	if (mixmax)
	{
		double best_rate = node->m_children[child_in_round[0]]->shot_wins / node->m_children[child_in_round[0]]->shot_po_count;
		wins += (best_rate)* budgetUsed;
		//update
		node->update_shot(budgetUsed, (1 - best_rate) * budgetUsed);
		return 1 - best_rate;
		
	}
	return node->m_children[child_in_round[0]].get_move();
	
}
std::vector<double> UCTSearch::think_hist(int color, passflag_t passflag)
{
	update_root();
	// set side to move
	m_rootstate.board.set_to_move(color);
	m_root->prepare_root_node(color, m_nodes, m_rootstate);
	std::vector<double> value_list;
	std::string value = "";
	std::fstream  file;
	file.open("out.txt", std::ios::app | std::ios::out);
	for (int i = 0; i < m_root->m_children.size(); i++)
	{
		auto nextstate = std::make_unique<GameState>(m_rootstate);
		int move = m_root->m_children[i]->get_move();
		if (move == -1)
			continue;
		nextstate->play_move(move);
		const auto raw_netlist = Network::get_scored_moves(
			nextstate.get(), Network::Ensemble::RANDOM_SYMMETRY);
		
		value_list.emplace_back(raw_netlist.winrate);
		//myprintf("%f,", raw_netlist.winrate);
	}
	int tpn = value_list.size();
	{
		for (int tmpi = 0; tmpi < tpn - 1; tmpi++) {
			for (int tmpj = 0; tmpj < tpn - tmpi - 1; tmpj++) {
				if (value_list[tmpj] > value_list[tmpj+1]) {
					double tempi = value_list[tmpj];
					value_list[tmpj] = value_list[tmpj + 1];
					value_list[tmpj + 1] = tempi;
				}
			}
		}
	}
	int count = 1;
	int enumc = 0;
	double sub_sum = 0;
	for (int tmpi = 0; tmpi < tpn ; tmpi++) {
		enumc++; 
		sub_sum += 1 - value_list[tmpi];
		if ((tmpi + 1) % count == 0) 
		{
			file << sub_sum/ count << ",";
			sub_sum = 0;
		}
	}
	file << "\n";
	file.close();
	return value_list;
}
int UCTSearch::think_shot(int color, passflag_t passflag,int bestmove,int coin,int poresmode,int pw,int mixmax, int width) {

	const auto raw_netlist = Network::get_scored_moves(
		&m_rootstate, Network::Ensemble::RANDOM_SYMMETRY);
	if (raw_netlist.winrate > 0.98)
		return -1;
	if (raw_netlist.winrate < 0.02)
		return -2;

	// Start counting time for us
	srand((unsigned)time(NULL));
	Random rd = Random(time(NULL));
	
	m_rootstate.start_clock(color);

	// set up timing info
	Time start;

	update_root();
	// set side to move
	m_rootstate.board.set_to_move(color);

	m_rootstate.get_timecontrol().set_boardsize(
		m_rootstate.board.get_boardsize());
	auto time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color, m_rootstate.get_movenum());

	m_root->prepare_root_node(color, m_nodes, m_rootstate);

	// create a sorted list of legal moves (make sure we
	// play something legal and decent even in time trouble)
	//m_root->prepare_root_node(color, m_nodes, m_rootstate);

	int budgetUsed = 0;
	int playouts = 0;
	double wins = 0;
	int resmove = shot(m_rootstate, m_root.get(), rd, coin, budgetUsed, playouts, wins, true, poresmode,pw,bestmove, mixmax, width);

	//m_last_rootstate = std::make_unique<GameState>(m_rootstate);
	return resmove;
}
int UCTSearch::policymove(int color, passflag_t passflag) {
	const auto raw_netlist = Network::get_scored_moves(
		&m_rootstate, Network::Ensemble::RANDOM_SYMMETRY);
	if (raw_netlist.winrate > 0.98)
		return -1;
	if (raw_netlist.winrate < 0.02)
		return -2;
	update_root();
	Random rd = Random(time(NULL));
	srand((unsigned)time(NULL));
	// set side to move
	m_rootstate.board.set_to_move(color);
	m_root->prepare_root_node(color, m_nodes, m_rootstate);
	return m_root->m_children[0].get_move();
	
}

int UCTSearch::valuemove(int color, passflag_t passflag) {
	const auto raw_netlist = Network::get_scored_moves(
		&m_rootstate, Network::Ensemble::RANDOM_SYMMETRY);
	if (raw_netlist.winrate > 0.98)
		return -1;
	if (raw_netlist.winrate < 0.02)
		return -2;
	update_root();
	Random rd = Random(time(NULL));
	srand((unsigned)time(NULL));
	// set side to move
	m_rootstate.board.set_to_move(color);
	m_root->prepare_root_node(color, m_nodes, m_rootstate);
	double best_value=1.0;
	int best_move=0;
	for (int i = 0; i < m_root->m_children.size(); i++)
	{
		auto nextstate = std::make_unique<GameState>(m_rootstate);
		int move = m_root->m_children[i]->get_move();
		if (move == -1)
			continue;
		nextstate->play_move(move);
		const auto raw_netlist = Network::get_scored_moves(
			nextstate.get(), Network::Ensemble::RANDOM_SYMMETRY);
		if (best_value > raw_netlist.winrate)
		{
			best_value = raw_netlist.winrate;
			best_move = i;
		}

	}
	 best_value = 1.0;
	 best_move = 0;
	for (int i = 0; i < 2; i++)
	{
		auto nextstate = std::make_unique<GameState>(m_rootstate);
		int move = m_root->m_children[i]->get_move();
		if (move == -1)
			continue;
		nextstate->play_move(move);
		const auto raw_netlist = Network::get_scored_moves(
			nextstate.get(), Network::Ensemble::RANDOM_SYMMETRY);
		if (best_value > raw_netlist.winrate)
		{
			best_value = raw_netlist.winrate;
			best_move = i;
		}

	}
	myprintf("bestmove %d,best_value:%f\n", best_move,1.0 - best_value);
	return  m_root->m_children[best_move]->get_move();
}
int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    m_rootstate.get_timecontrol().set_boardsize(
        m_rootstate.board.get_boardsize());
	auto time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(color, m_nodes, m_rootstate);
	/*
	for (int tmpj = 0; tmpj < m_root->m_children.size(); tmpj++)
	{
		std::string vertex = m_rootstate.move_to_text(m_root->m_children[tmpj].get_move());

		myprintf("%s\t%f\t%f\t%f\t%d\n",
			vertex.c_str(),
			m_root->m_children[tmpj]->get_score(),
			m_root->m_children[tmpj]->get_eval(color),
			m_root->m_children[tmpj]->m_net_eval,
			m_root->m_children[tmpj]->get_visits());
		//rp_count[tmpj], rp_win[tmpj]);
	}
	*/
    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    bool keeprunning = true;
    int last_update = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        // output some stats every few seconds
        // check if we should still search
        if (elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            dump_analysis(static_cast<int>(m_playouts));
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
    } while (keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();

    // reactivate all pruned root children
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // display search info
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);
    Training::record(m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    if (elapsed_centis+1 > 0) {
        myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
                 m_root->get_visits(),
                 static_cast<int>(m_nodes),
                 static_cast<int>(m_playouts),
                 (m_playouts * 100.0) / (elapsed_centis+1));
    }
    int bestmove = get_best_move(passflag);
	/*
	for (int tmpj = 0; tmpj < m_root->m_children.size(); tmpj++)
	{
		std::string vertex = m_rootstate.move_to_text(m_root->m_children[tmpj].get_move());

		myprintf("%s\t%f\t%f\t%f\t%d\n",
			vertex.c_str(),
			m_root->m_children[tmpj]->get_score(),
			m_root->m_children[tmpj]->get_eval(color),
			m_root->m_children[tmpj]->m_net_eval,
			m_root->m_children[tmpj]->get_visits());
		//rp_count[tmpj], rp_win[tmpj]);
	}
	*/
    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}

void UCTSearch::ponder() {
    update_root();

    m_root->prepare_root_node(m_rootstate.board.get_to_move(),
                              m_nodes, m_rootstate);

    m_run = true;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cfg_num_threads; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }
    auto keeprunning = true;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(0, 1);
    } while (!Utils::input_pending() && keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();

    // display search info
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    m_maxplayouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    // Limit to type max / 2 to prevent overflow when multithreading.
    m_maxvisits = std::min(visits, UNLIMITED_PLAYOUTS);
}
