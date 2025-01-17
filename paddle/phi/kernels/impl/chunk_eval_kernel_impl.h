// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <set>
#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/utils/optional.h"

namespace phi {
struct Segment {
  int begin;
  int end;
  int type;
  bool operator==(const Segment& y) const {
    return begin == y.begin && end == y.end && type == y.type;
  }
};

bool ChunkEnd(int prev_tag,
              int prev_type,
              int tag,
              int type,
              int other_chunk_type,
              int tag_begin,
              int tag_inside,
              int tag_end,
              int tag_single);

bool ChunkBegin(int prev_tag,
                int prev_type,
                int tag,
                int type,
                int other_chunk_type,
                int tag_begin,
                int tag_inside,
                int tag_end,
                int tag_single);

void EvalOneSeq(const int64_t* output,
                const int64_t* label,
                int length,
                std::vector<Segment>* output_segments,
                std::vector<Segment>* label_segments,
                int64_t* num_output_segments,
                int64_t* num_label_segments,
                int64_t* num_correct,
                int num_chunk_types,
                int num_tag_types,
                int other_chunk_type,
                int tag_begin,
                int tag_inside,
                int tag_end,
                int tag_single,
                const std::set<int>& excluded_chunk_types);

void GetSegments(const int64_t* label,
                 int length,
                 std::vector<Segment>* segments,
                 int num_chunk_types,
                 int num_tag_types,
                 int other_chunk_type,
                 int tag_begin,
                 int tag_inside,
                 int tag_end,
                 int tag_single) {
  segments->clear();
  segments->reserve(length);
  int chunk_start = 0;
  bool in_chunk = false;
  int tag = -1;
  int type = other_chunk_type;
  for (int i = 0; i < length; ++i) {
    int prev_tag = tag;
    int prev_type = type;
    PADDLE_ENFORCE_LE(
        label[i],
        num_chunk_types * num_tag_types,
        phi::errors::InvalidArgument(
            "The value of Input(Label) should be less than the number of "
            "chunk types times the number of tag types, but received %d "
            "(Label) vs %d (chunk types) * %d (tag types).",
            label[i],
            num_chunk_types,
            num_tag_types));
    tag = label[i] % num_tag_types;
    type = label[i] / num_tag_types;
    if (in_chunk && ChunkEnd(prev_tag,
                             prev_type,
                             tag,
                             type,
                             other_chunk_type,
                             tag_begin,
                             tag_inside,
                             tag_end,
                             tag_single)) {
      Segment segment{
          chunk_start,  // begin
          i - 1,        // end
          prev_type,
      };
      segments->push_back(segment);
      in_chunk = false;
    }
    if (ChunkBegin(prev_tag,
                   prev_type,
                   tag,
                   type,
                   other_chunk_type,
                   tag_begin,
                   tag_inside,
                   tag_end,
                   tag_single)) {
      chunk_start = i;
      in_chunk = true;
    }
  }
  if (in_chunk) {
    Segment segment{
        chunk_start,  // begin
        length - 1,   // end
        type,
    };
    segments->push_back(segment);
  }
}

bool ChunkEnd(int prev_tag,
              int prev_type,
              int tag,
              int type,
              int other_chunk_type,
              int tag_begin,
              int tag_inside,
              int tag_end,
              int tag_single) {
  if (prev_type == other_chunk_type) return false;
  if (type == other_chunk_type) return true;
  if (type != prev_type) return true;
  if (prev_tag == tag_begin) return tag == tag_begin || tag == tag_single;
  if (prev_tag == tag_inside) return tag == tag_begin || tag == tag_single;
  if (prev_tag == tag_end) return true;
  if (prev_tag == tag_single) return true;
  return false;
}

bool ChunkBegin(int prev_tag,
                int prev_type,
                int tag,
                int type,
                int other_chunk_type,
                int tag_begin,
                int tag_inside,
                int tag_end,
                int tag_single) {
  if (prev_type == other_chunk_type) return type != other_chunk_type;
  if (type == other_chunk_type) return false;
  if (type != prev_type) return true;
  if (tag == tag_begin) return true;
  if (tag == tag_inside) return prev_tag == tag_end || prev_tag == tag_single;
  if (tag == tag_end) return prev_tag == tag_end || prev_tag == tag_single;
  if (tag == tag_single) return true;
  return false;
}

template <typename T, typename Context>
void ChunkEvalKernel(const Context& dev_ctx,
                     const DenseTensor& inference,
                     const DenseTensor& label,
                     const paddle::optional<DenseTensor>& seq_length,
                     int num_chunk_types,
                     const std::string& chunk_scheme,
                     const std::vector<int>& excluded_chunk_types,
                     DenseTensor* precision,
                     DenseTensor* recall,
                     DenseTensor* f1_score,
                     DenseTensor* num_infer_chunks,
                     DenseTensor* num_label_chunks,
                     DenseTensor* num_correct_chunks) {
  // initialize to parse configurations
  int num_tag_types;
  int other_chunk_type;
  int tag_begin, tag_inside, tag_end, tag_single;
  std::vector<Segment> label_segments;
  std::vector<Segment> output_segments;
  std::set<int> excluded_chunk_types_new;

  if (chunk_scheme == "IOB") {
    num_tag_types = 2;
    tag_begin = 0;
    tag_inside = 1;
    tag_end = -1;
    tag_single = -1;
  } else if (chunk_scheme == "IOE") {
    num_tag_types = 2;
    tag_begin = -1;
    tag_inside = 0;
    tag_end = 1;
    tag_single = -1;
  } else if (chunk_scheme == "IOBES") {
    num_tag_types = 4;
    tag_begin = 0;
    tag_inside = 1;
    tag_end = 2;
    tag_single = 3;
  } else if (chunk_scheme == "plain") {
    num_tag_types = 1;
    tag_begin = -1;
    tag_inside = -1;
    tag_end = -1;
    tag_single = -1;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unknown chunk scheme."));
  }
  other_chunk_type = num_chunk_types;
  excluded_chunk_types_new.insert(excluded_chunk_types.begin(),
                                  excluded_chunk_types.end());

  const int64_t* inference_data = inference.data<int64_t>();
  const int64_t* label_data = label.data<int64_t>();
  T* precision_data = dev_ctx.template Alloc<T>(precision);
  T* recall_data = dev_ctx.template Alloc<T>(recall);
  T* f1_data = dev_ctx.template Alloc<T>(f1_score);
  int64_t* num_infer_chunks_data =
      dev_ctx.template Alloc<int64_t>(num_infer_chunks);
  int64_t* num_label_chunks_data =
      dev_ctx.template Alloc<int64_t>(num_label_chunks);
  int64_t* num_correct_chunks_data =
      dev_ctx.template Alloc<int64_t>(num_correct_chunks);
  *num_infer_chunks_data = 0;
  *num_label_chunks_data = 0;
  *num_correct_chunks_data = 0;

  auto lod = label.lod();
  bool use_padding = lod.empty();
  int num_sequences = 0;

  if (use_padding) {
    auto dim1 = inference.dims()[1];
    auto* seq_length_t = seq_length.get_ptr();
    auto* seq_length_data = seq_length_t->data<int64_t>();
    num_sequences = seq_length_t->dims()[0];

    for (int i = 0; i < num_sequences; ++i) {
      int seq_length = seq_length_data[i];
      EvalOneSeq(inference_data + i * dim1,
                 label_data + i * dim1,
                 seq_length,
                 &output_segments,
                 &label_segments,
                 num_infer_chunks_data,
                 num_label_chunks_data,
                 num_correct_chunks_data,
                 num_chunk_types,
                 num_tag_types,
                 other_chunk_type,
                 tag_begin,
                 tag_inside,
                 tag_end,
                 tag_single,
                 excluded_chunk_types_new);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        lod.size(),
        1UL,
        phi::errors::InvalidArgument(
            "Only support one level LoD sequence now, but received %d.",
            lod.size()));
    PADDLE_ENFORCE_EQ(
        lod,
        inference.lod(),
        phi::errors::InvalidArgument(
            "Input(Inference) and Input(Label) of Op(chunk_eval) should have "
            "same LoD information."));
    num_sequences = lod[0].size() - 1;

    for (int i = 0; i < num_sequences; ++i) {
      int seq_length = lod[0][i + 1] - lod[0][i];
      EvalOneSeq(inference_data + lod[0][i],
                 label_data + lod[0][i],
                 seq_length,
                 &output_segments,
                 &label_segments,
                 num_infer_chunks_data,
                 num_label_chunks_data,
                 num_correct_chunks_data,
                 num_chunk_types,
                 num_tag_types,
                 other_chunk_type,
                 tag_begin,
                 tag_inside,
                 tag_end,
                 tag_single,
                 excluded_chunk_types_new);
    }
  }

  *precision_data =
      !(*num_infer_chunks_data)
          ? 0
          : static_cast<T>(*num_correct_chunks_data) / (*num_infer_chunks_data);
  *recall_data =
      !(*num_label_chunks_data)
          ? 0
          : static_cast<T>(*num_correct_chunks_data) / (*num_label_chunks_data);
  *f1_data = !(*num_correct_chunks_data)
                 ? 0
                 : 2 * (*precision_data) * (*recall_data) /
                       ((*precision_data) + (*recall_data));
}

void EvalOneSeq(const int64_t* output,
                const int64_t* label,
                int length,
                std::vector<Segment>* output_segments,
                std::vector<Segment>* label_segments,
                int64_t* num_output_segments,
                int64_t* num_label_segments,
                int64_t* num_correct,
                int num_chunk_types,
                int num_tag_types,
                int other_chunk_type,
                int tag_begin,
                int tag_inside,
                int tag_end,
                int tag_single,
                const std::set<int>& excluded_chunk_types) {
  GetSegments(output,
              length,
              output_segments,
              num_chunk_types,
              num_tag_types,
              other_chunk_type,
              tag_begin,
              tag_inside,
              tag_end,
              tag_single);
  GetSegments(label,
              length,
              label_segments,
              num_chunk_types,
              num_tag_types,
              other_chunk_type,
              tag_begin,
              tag_inside,
              tag_end,
              tag_single);
  size_t i = 0, j = 0;
  while (i < output_segments->size() && j < label_segments->size()) {
    if (output_segments->at(i) == label_segments->at(j) &&
        excluded_chunk_types.count(output_segments->at(i).type) != 1) {
      ++(*num_correct);
    }
    if (output_segments->at(i).end < label_segments->at(j).end) {
      ++i;
    } else if (output_segments->at(i).end > label_segments->at(j).end) {
      ++j;
    } else {
      ++i;
      ++j;
    }
  }
  for (auto& segment : (*label_segments)) {
    if (excluded_chunk_types.count(segment.type) != 1) {
      ++(*num_label_segments);
    }
  }
  for (auto& segment : (*output_segments)) {
    if (excluded_chunk_types.count(segment.type) != 1) {
      ++(*num_output_segments);
    }
  }
}
}  // namespace phi
