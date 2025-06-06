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
// limitations under the License.

#include "litert/core/model/buffer_manager.h"

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/test/matchers.h"

namespace litert::internal {

namespace {

static constexpr absl::string_view kData = "foo";

TEST(BufferManagerTest, EmptyFirstBuffer) {
  BufferManager manager;

  EXPECT_EQ(manager.NumBuffers(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffer,
                              manager.GetBuffer(BufferManager::kEmptyBufferId));
  EXPECT_EQ(buffer.Size(), 0);
}

TEST(BufferManagerTest, RegisterNonOwnedBuffer) {
  BufferManager manager;

  OwningBufferRef<uint8_t> buffer(kData);
  const auto id = manager.RegisterNonOwnedBuffer(buffer);

  EXPECT_EQ(manager.NumBuffers(), 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto managed_buffer, manager.GetBuffer(id));
  EXPECT_EQ(managed_buffer.StrView(), kData);
}

TEST(BufferManagerTest, RegisterOwnedBuffer) {
  BufferManager manager;

  OwningBufferRef<uint8_t> buffer(kData);
  const auto id = manager.RegisterOwnedBuffer(std::move(buffer));

  EXPECT_EQ(manager.NumBuffers(), 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto managed_buffer, manager.GetBuffer(id));
  EXPECT_EQ(managed_buffer.StrView(), kData);
}

TEST(BufferManagerTest, RegisterWithContext) {
  BufferManager manager;

  OwningBufferRef<uint8_t> buffer(kData);
  BufferContext context = {true};
  const auto id = manager.RegisterNonOwnedBuffer(buffer, context);

  EXPECT_EQ(manager.NumBuffers(), 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto managed_buffer, manager.GetBuffer(id));
  EXPECT_EQ(managed_buffer.StrView(), kData);
  LITERT_ASSERT_OK_AND_ASSIGN(auto managed_context, manager.GetContext(id));
  EXPECT_EQ(managed_context.get().should_append, true);
}

}  // namespace

}  // namespace litert::internal
