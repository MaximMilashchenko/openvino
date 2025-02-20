// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id roi_pooling::type_id() {
    static primitive_type_base<roi_pooling> instance;
    return &instance;
}

layout roi_pooling_inst::calc_output_layout(roi_pooling_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for roi_pooling_node!");
    auto desc = node.get_primitive();
    layout data_layout = node.input().get_output_layout();
    layout rois_layout = node.rois().get_output_layout();
    int num_rois = rois_layout.batch();
    int out_fm = desc->position_sensitive ? desc->output_dim : data_layout.feature();

    return layout(data_layout.data_type, format::bfyx, {num_rois, out_fm, desc->pooled_width, desc->pooled_height});
}

std::string roi_pooling_inst::to_string(roi_pooling_node const& node) {
    auto desc = node.get_primitive();
    auto mode = desc->mode == pooling_mode::max
                    ? "max"
                    : desc->mode == pooling_mode::bilinear
                          ? "bilinear"
                          : desc->mode == pooling_mode::deformable_bilinear ? "deformable_bilinear" : "average";
    auto is_ps = desc->position_sensitive ? "true" : "false";
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite roi_info;
    roi_info.add("mode", mode);
    roi_info.add("position sensitive", is_ps);
    roi_info.add("pooled_w", desc->pooled_width);
    roi_info.add("pooled_h", desc->pooled_height);
    roi_info.add("spatial_scale", desc->spatial_scale);
    roi_info.add("output_dim", desc->output_dim);
    roi_info.add("spatial_bins_x", desc->spatial_bins_x);
    roi_info.add("spatial_bins_y", desc->spatial_bins_y);
    roi_info.add("trans_std", desc->trans_std);
    roi_info.add("no_trans", desc->no_trans);
    roi_info.add("part_size", desc->part_size);

    node_info->add("roi info", roi_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
