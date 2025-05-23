// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <list>
#include <stack>
#include <unordered_set>
#include <vector>

namespace qnn {

template <typename Op>
struct Node {
  Op *op = nullptr;
  std::list<Node<Op> *> nodes;
};

template <typename Op>
bool IsMatch(Op *lhs, Op *rhs);

template <typename Op>
bool FullyMatchImpl(Node<Op> *graph, Node<Op> *pattern,
                    std::unordered_set<Node<Op> *> &graph_visited,
                    std::unordered_set<Node<Op> *> &pattern_visited) {
  if (!IsMatch(graph->op, pattern->op)) {
    return false;
  }

  graph_visited.emplace(graph);
  pattern_visited.emplace(pattern);

  auto graph_it = graph->nodes.begin();
  auto pattern_it = pattern->nodes.begin();
  while(true) {
    while (graph_it != graph->nodes.end() && graph_visited.count(*graph_it)) {
        ++graph_it;
    }
    while (pattern_it != pattern->nodes.end() && pattern_visited.count(*pattern_it)) {
        ++pattern_it;
    }

    if (graph_it == graph->nodes.end() || pattern_it == pattern->nodes.end()){
        break;
    }

    if (FullyMatchImpl(*graph_it, *pattern_it, graph_visited, pattern_visited)) {
        ++graph_it;
        ++pattern_it;
    }
    else {
        ++graph_it;
    }
  }

  return pattern_it == pattern->nodes.end();
}

template <typename Op>
bool FullyMatch(Node<Op> *graph, Node<Op> *pattern) {
  if (graph == nullptr || pattern == nullptr) {
    return false;
  }

  std::unordered_set<Node<Op> *> visited;

  std::stack<Node<Op> *> pattern_start;
  pattern_start.emplace(graph);
  while (!pattern_start.empty()) {
    auto *top = pattern_start.top();
    pattern_start.pop();

    std::unordered_set<Node<Op> *> graph_visited;
    std::unordered_set<Node<Op> *> pattern_visited;
    if (FullyMatchImpl(top, pattern, graph_visited, pattern_visited)) {
        // find one match
        return true;
    }
    visited.emplace(top);

    for (auto *ptr : top->nodes) {
        if (auto it = visited.find(ptr); it != visited.end()) {
            continue;
        }

        pattern_start.emplace(ptr);
    }
  }

  return false;
}

}  // namespace qnn