// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and

#include "litert/vendors/qualcomm/qnn_manager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/tools/dump.h"

namespace {

using ::litert::qnn::QnnManager;
using ::litert::qnn::internal::Dump;
using ::testing::HasSubstr;

// NOTE: This tests that all of the dynamic loading works properly and
// the QNN SDK instance can be properly initialized and destroyed.

TEST(QnnManagerTest, SetupQnnManager) {
  auto configs = QnnManager::DefaultBackendConfigs();
  auto options = ::qnn::Options();
  auto qnn = QnnManager::Create(configs, options);
  ASSERT_TRUE(qnn);
}

TEST(QnnManagerTest, Dump) {
  auto configs = QnnManager::DefaultBackendConfigs();
  auto options = ::qnn::Options();
  auto qnn = QnnManager::Create(configs, options);
  ASSERT_TRUE(qnn);

  auto dump = Dump(**qnn);

  EXPECT_THAT(dump, HasSubstr("< QnnInterface_t >"));
  EXPECT_THAT(dump, HasSubstr("< QnnSystemInterface_t >"));
}

}  // namespace
