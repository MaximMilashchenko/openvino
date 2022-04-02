// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/topk.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
// Had to split out these two functions. They used to be lambda expressions but
// MSVC had difficulty compiling. This way is more explicit.
template <typename T, typename U>
inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
// this is intentional to be able to compare floats directly
// without using relative or absolute tolerance
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (std::get<0>(a) == std::get<0>(b)) {
        return std::get<1>(a) < std::get<1>(b);
    }
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
    return a > b;
}

template <typename T, typename U>
inline bool compare_min(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    return a < b;
}

template <typename T, typename U>
inline bool sort_indices_ascending(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    return std::get<1>(a) < std::get<1>(b);
}

// my sorting
template <class RandomAccessIterator, class Compare>
void insertionsort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
    if (first == last) return;

    for (RandomAccessIterator it = first+1; it < last; ++it)
    {
        RandomAccessIterator current_element = it;

        for (RandomAccessIterator i = it - 1; i >= first; --i)
        {
            if (comp(*i, *current_element)) 
            {
                std::swap(*i, *current_element);
                current_element--;
            } 
            else 
            {
                break;
            }
        }
    }
}

template <class RandomAccessIterator, class Compare>
RandomAccessIterator Partition(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
    RandomAccessIterator pivotal = std::prev(last, 1);
    RandomAccessIterator less = first;

    for (RandomAccessIterator j = first; j < pivotal; ++j) 
    {
        if (comp(*j, *pivotal))
        {
            swap(*j, *less);
            ++less; 
        }
    }
    std::swap(*less, *pivotal);
    return less;
}

template <class RandomAccessIterator, class Compare>
void _sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
    if (std::distance(first, last) < 10)
    {
        insertionsort(first, last, comp);
    }  
    else if (std::distance(first, last) > 1)
    {
        RandomAccessIterator q = Partition(first, last, comp);
        _sort(first, q, comp);
        _sort(q++, last, comp);
    }
}

template <typename T, typename U>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const Shape& in_shape,
          const Shape& out_shape,
          size_t axis,
          size_t k,
          bool compute_max,
          op::v1::TopK::SortType sort = op::v1::TopK::SortType::NONE) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    using namespace std;
    // reorder source axis visit order and make "axis" inner most
    size_t ndim = static_cast<size_t>(in_shape.size());
    Coordinate start_corner(ndim, 0);
    Coordinate end_corner(in_shape);
    end_corner[axis] = 1;
    Strides strides(ndim, 1);
    AxisVector axis_order(ndim);
    iota(axis_order.begin(), axis_order.end(), 0);
    axis_order.erase(axis_order.begin() + axis);
    axis_order.push_back(axis);
    // Create CoordinateTransforms that visits only the first element along "axis"
    CoordinateTransform input_transform(in_shape, start_corner, end_corner, strides, axis_order);
    CoordinateTransform output_transform(out_shape, start_corner, end_corner, strides, axis_order);
    // Create temp vector for sorting.
    vector<tuple<T, U>> workspace(in_shape[axis]);
    vector<size_t> in_strides = ngraph::row_major_strides(in_shape);
    vector<size_t> out_strides = ngraph::row_major_strides(out_shape);
    auto in_axis_stride = in_strides[axis];
    auto out_axis_stride = out_strides[axis];
    for (const Coordinate& coord : input_transform) {
        auto arg_index = input_transform.index(coord);
        auto out_index = output_transform.index(coord);
        // Fill the temp vector
        U i = 0;
        for (tuple<T, U>& entry : workspace) {
            get<0>(entry) = arg[arg_index];
            get<1>(entry) = i;
            arg_index += in_axis_stride;
            i++;
        }
        // Sort the temp vector
        if (compute_max) {
            nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), compare_max<T, U>);
        } else {
            nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), compare_min<T, U>);
        }
        // Write temp vector to output
        switch (sort) {
        case op::v1::TopK::SortType::NONE:
            break;
        case op::v1::TopK::SortType::SORT_INDICES:
            _sort(workspace.begin(), workspace.begin() + k, sort_indices_ascending<T, U>);
            break;
        case op::v1::TopK::SortType::SORT_VALUES:
            if (compute_max)
                _sort(workspace.begin(), workspace.begin() + k, compare_max<T, U>);
            else
                _sort(workspace.begin(), workspace.begin() + k, compare_min<T, U>);
        }
        for (size_t j = 0; j < k; j++) {
            tuple<T, U> entry = workspace[j];
            out_values[out_index] = get<0>(entry);
            out_indices[out_index] = get<1>(entry);
            out_index += out_axis_stride;
        }
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
