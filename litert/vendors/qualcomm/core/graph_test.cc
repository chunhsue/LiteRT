#include "litert/vendors/qualcomm/core/graph.h"

#include <gtest/gtest.h>

#include "QnnOpDef.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

template <>
bool IsMatch(OpWrapper *lhs, OpWrapper *rhs) {
  return lhs->GetOpCode() == rhs->GetOpCode();
}

namespace {

TEST(GraphTest, Init) {
  Node<OpWrapper> node;
  EXPECT_EQ(node.op, nullptr);
  EXPECT_TRUE(node.nodes.empty());
}

TEST(GraphTest, SingleOp) {
  OpWrapper op("", QNN_OP_ARGMAX, QnnOpCode::kArgmin);
  Node<OpWrapper> node;
  node.op = &op;

  EXPECT_TRUE(FullyMatch(&node, &node));
}

TEST(GraphTest, MultipleOp) {
  OpWrapper op3("", QNN_OP_CAST, QnnOpCode::kCast);
  OpWrapper op2("", QNN_OP_ARGMIN, QnnOpCode::kArgmin);
  OpWrapper op1("", QNN_OP_ARGMAX, QnnOpCode::kArgmax);
  Node<OpWrapper> node3;
  node3.op = &op3;
  Node<OpWrapper> node2;
  node2.op = &op2;
  node2.nodes.emplace_back(&node3);
  Node<OpWrapper> node1;
  node1.op = &op1;
  node1.nodes.emplace_back(&node2);

  Node<OpWrapper> pattern_node2;
  pattern_node2.op = &op3;
  Node<OpWrapper> pattern_node1;
  pattern_node1.op = &op2;
  pattern_node1.nodes.emplace_back(&pattern_node2);

  EXPECT_TRUE(FullyMatch(&node1, &pattern_node1));
}

TEST(GraphTest, MultipleOpNotMatch) {
  OpWrapper op3("", QNN_OP_CAST, QnnOpCode::kCast);
  OpWrapper op2("", QNN_OP_ARGMIN, QnnOpCode::kArgmin);
  OpWrapper op1("", QNN_OP_ARGMAX, QnnOpCode::kArgmax);
  Node<OpWrapper> node3;
  node3.op = &op3;
  Node<OpWrapper> node2;
  node2.op = &op2;
  node2.nodes.emplace_back(&node3);
  Node<OpWrapper> node1;
  node1.op = &op1;
  node1.nodes.emplace_back(&node2);
  
  OpWrapper pattern_op2("", QNN_OP_CONV_2D, QnnOpCode::kConv2d);
  OpWrapper pattern_op1("", QNN_OP_CAST, QnnOpCode::kCast);
  Node<OpWrapper> pattern_node2;
  pattern_node2.op = &pattern_op2;
  Node<OpWrapper> pattern_node1;
  pattern_node1.op = &pattern_op1;
  pattern_node1.nodes.emplace_back(&pattern_node2);

  EXPECT_FALSE(FullyMatch(&node1, &pattern_node1));
}

}  // namespace
}  // namespace qnn