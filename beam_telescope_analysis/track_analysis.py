''' Track finding and fitting functions are listed here.'''
from __future__ import division

import logging
import os.path
from multiprocessing import Pool, cpu_count
import math
from collections import Iterable
import functools
import itertools
from itertools import combinations

import tables as tb
import numpy as np
from numba import njit
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm

from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools import plot_utils
# from beam_telescope_analysis.tools import smc
from beam_telescope_analysis.tools import analysis_utils
from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.tools import kalman


def find_tracks(telescope_configuration, input_merged_file, output_track_candidates_file=None, select_extrapolation_duts=None, align_to_beam=True, max_events=None, chunk_size=1000000):
    '''Sorting DUT hits and tries to find hits in subsequent DUTs matching the hits in the first DUT.
    The output is the track candidates array which has the hits in a different order compared to the tracklets array (merged array).

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_merged_file : string
        Filename of the input merged cluster file containing the hit information from all DUTs.
    output_track_candidates_file : string
        Filename of the output track candidates file.
    select_extrapolation_duts : list
        The given DUTs will be used for track extrapolation for improving track finding efficiency.
        In some rare cases, removing DUTs with a coarse resolution might improve track finding efficiency.
        If None, select all DUTs.
        If list is empty or has a single entry, disable extrapolation (at least 2 DUTs are required for extrapolation to work).
    align_to_beam : bool
        If True, the telescope alignment is used to align the DUTs so that the beam axis is parallel to the z axis.
        This improves the performance of track finding algorithm.
        If False, the beam axis is not corrected and large track angles and high track densities
        have an impact on the performance of the track finding algorithm.
    max_events : uint
        Maximum number of randomly chosen events. If None, all events are taken.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_track_candidates_file : string
        Filename of the output track candidates file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Finding tracks in %d DUTs ===' % n_duts)

    if output_track_candidates_file is None:
        output_track_candidates_file = os.path.join(os.path.dirname(input_merged_file), 'Track_Candidates.h5')

    if select_extrapolation_duts is None:
        select_extrapolation_duts = list(range(n_duts))
    elif isinstance(select_extrapolation_duts, (list, tuple)):
        if set(select_extrapolation_duts) - set(range(n_duts)):
            raise ValueError("Found invalid DUTs in select_extrapolation_duts: %s" % ', '.join(str(dut) for dut in (set(select_extrapolation_duts) - set(range(n_duts)))))
    else:
        raise ValueError("Invalid value for select_extrapolation_duts")

# TODO: implement max_events into SMC
#     def work(tracklets_data_chunk):
#         ''' Track finding per cpu core '''
#         # Prepare hit data for track finding, create temporary arrays for x, y, z position and charge data
#         # This is needed to call a numba jitted function, since the number of DUTs is not fixed and thus the data format
#         x_local = np.copy(tracklets_data_chunk['x_dut_0'])
#         y_local = np.copy(tracklets_data_chunk['y_dut_0'])
#         z_local = np.copy(tracklets_data_chunk['z_dut_0'])
#         x_err_local = np.copy(tracklets_data_chunk['x_err_dut_0'])
#         y_err_local = np.copy(tracklets_data_chunk['y_err_dut_0'])
#         z_err_local = np.copy(tracklets_data_chunk['z_err_dut_0'])
#         geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=0, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
#         x = tracklets_data_chunk['x_dut_0']
#         y = tracklets_data_chunk['y_dut_0']
#         z = tracklets_data_chunk['z_dut_0']
#         x_err = tracklets_data_chunk['x_err_dut_0']
#         y_err = tracklets_data_chunk['y_err_dut_0']
#         z_err = tracklets_data_chunk['z_err_dut_0']
#         charge = tracklets_data_chunk['charge_dut_0']
#         n_hits = tracklets_data_chunk['n_hits_dut_0']
#         cluster_shape = tracklets_data_chunk['cluster_shape_dut_0']
#         n_cluster = tracklets_data_chunk['n_cluster_dut_0']
#         for dut_index in range(1, n_duts):
#             x_local = np.column_stack((x_local, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
#             y_local = np.column_stack((y_local, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
#             z_local = np.column_stack((z_local, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
#             x_err_local = np.column_stack((x_err_local, tracklets_data_chunk['x_err_dut_%d' % (dut_index)]))
#             y_err_local = np.column_stack((y_err_local, tracklets_data_chunk['y_err_dut_%d' % (dut_index)]))
#             z_err_local = np.column_stack((z_err_local, tracklets_data_chunk['z_err_dut_%d' % (dut_index)]))
#             geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
#             x = np.column_stack((x, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
#             y = np.column_stack((y, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
#             z = np.column_stack((z, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
#             x_err = np.column_stack((x_err, tracklets_data_chunk['x_err_dut_%d' % (dut_index)]))
#             y_err = np.column_stack((y_err, tracklets_data_chunk['y_err_dut_%d' % (dut_index)]))
#             z_err = np.column_stack((z_err, tracklets_data_chunk['z_err_dut_%d' % (dut_index)]))
#             charge = np.column_stack((charge, tracklets_data_chunk['charge_dut_%d' % (dut_index)]))
#             n_hits = np.column_stack((n_hits, tracklets_data_chunk['n_hits_dut_%d' % (dut_index)]))
#             cluster_shape = np.column_stack((cluster_shape, tracklets_data_chunk['cluster_shape_dut_%d' % (dut_index)]))
#             n_cluster = np.column_stack((n_cluster, tracklets_data_chunk['n_cluster_dut_%d' % (dut_index)]))

#         event_number = tracklets_data_chunk['event_number']
#         hit_flag = np.zeros_like(tracklets_data_chunk['hit_flag'])
#         quality_flag = np.zeros_like(tracklets_data_chunk['quality_flag'])
#         n_tracks = tracklets_data_chunk['n_tracks']

#         # Perform the track finding with jitted loop
#         _find_tracks_loop(event_number=event_number,
#                           x_local=x_local,
#                           y_local=y_local,
#                           z_local=z_local,
#                           x_err_local=x_err_local,
#                           y_err_local=y_err_local,
#                           z_err_local=z_err_local,
#                           x=x,
#                           y=y,
#                           z=z,
#                           x_err=x_err,
#                           y_err=y_err,
#                           z_err=z_err,
#                           charge=charge,
#                           n_hits=n_hits,
#                           cluster_shape=cluster_shape,
#                           n_cluster=n_cluster,
#                           hit_flag=hit_flag,
#                           n_tracks=n_tracks)

# # TODO: also use local coordinates in find_tracks_loop to avoid transformation to local coordinate system
# #         for dut_index in range(0, n_duts):
# #             geometry_utils.apply_alignment_to_hits(hits=combined, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, inverse=True, no_z=False)

#         # Merge result data from arrays into one recarray
#         combined = np.column_stack((event_number, x_local, y_local, z_local, charge, n_hits, cluster_shape, n_cluster, hit_flag, quality_flag, n_tracks, x_err_local, y_err_local, z_err_local))
#         return np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

#     smc.SMC(input_filename=input_tracklets_file,
#             output_filename=output_track_candidates_file,
#             func=work,
#             node_desc={'name':'TrackCandidates',
#                         'title':'Track candidates'},
#             # Apply track finding on tracklets or track candidates
#             table=['Tracklets', 'TrackCandidates'],
#             align_at='event_number',
#             chunk_size=chunk_size)

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in telescope:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)

    with tb.open_file(input_merged_file, mode='r') as in_file_h5:
        tracklets_node = in_file_h5.root.MergedClusters

        with tb.open_file(output_track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(
                where=out_file_h5.root, name='TrackCandidates',
                description=tracklets_node.dtype,
                title='Track candidates',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

            total_n_tracks = tracklets_node.shape[0]
            total_n_tracks_stored = 0
            total_n_events_stored = 0

            progress_bar = tqdm(total=total_n_tracks, ncols=80)

            for tracklets_data_chunk, index_chunk in analysis_utils.data_aligned_at_events(tracklets_node, chunk_size=chunk_size):
                n_tracks_chunk = tracklets_data_chunk.shape[0]

                unique_events = np.unique(tracklets_data_chunk["event_number"])
                n_events_chunk = unique_events.shape[0]

                if max_events:
                    if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                        select_n_events = max_events - total_n_events_stored
                    elif total_n_events_stored == 0:  # first chunk
                        select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                    else:
                        # calculate correction of number of selected events
                        correction = (total_n_tracks - index_chunk) / total_n_tracks * 1 / (((total_n_tracks - last_index_chunk) / total_n_tracks) / ((max_events - total_n_events_stored_last) / max_events)) \
                            + (index_chunk) / total_n_tracks * 1 / (((last_index_chunk) / total_n_tracks) / ((total_n_events_stored_last) / max_events))
                        select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                    # do not store more events than in current chunk
                    select_n_events = min(n_events_chunk, select_n_events)
                    # do not store more events than given by max_events
                    select_n_events = min(select_n_events, max_events - total_n_events_stored)
                    np.random.seed(seed=0)
                    selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                    store_n_events = selected_events.shape[0]
                    total_n_events_stored += store_n_events
                    selected_tracks = np.in1d(tracklets_data_chunk["event_number"], selected_events)
                    store_n_tracks = np.count_nonzero(selected_tracks)
                    total_n_tracks_stored += store_n_tracks
                    tracklets_data_chunk = tracklets_data_chunk[selected_tracks]
                indices = np.column_stack([np.arange(tracklets_data_chunk.shape[0], dtype=np.int64) for _ in range(n_duts)])
                event_numbers = tracklets_data_chunk['event_number']
                x_global = []
                y_global = []
                z_global = []
                translation = []
                rotation = []
                normal = []
                for dut_index, dut in enumerate(telescope):
                    x_global_dut, y_global_dut, z_global_dut = dut.local_to_global_position(
                        x=tracklets_data_chunk['x_dut_%d' % dut_index],
                        y=tracklets_data_chunk['y_dut_%d' % dut_index],
                        z=tracklets_data_chunk['z_dut_%d' % dut_index])
                    if align_to_beam:
                        x_global_dut, y_global_dut, z_global_dut = dut.local_to_global_position(
                            x=x_global_dut,
                            y=y_global_dut,
                            z=z_global_dut,
                            translation_x=telescope.translation_x,
                            translation_y=telescope.translation_y,
                            translation_z=telescope.translation_z,
                            rotation_alpha=telescope.rotation_alpha,
                            rotation_beta=telescope.rotation_beta,
                            rotation_gamma=telescope.rotation_gamma)
                    x_global.append(x_global_dut)
                    y_global.append(y_global_dut)
                    z_global.append(z_global_dut)
                    translation.append([dut.translation_x, dut.translation_y, dut.translation_z])
                    dut_rotation_matrix = geometry_utils.rotation_matrix(
                        alpha=dut.rotation_alpha,
                        beta=dut.rotation_beta,
                        gamma=dut.rotation_gamma)
                    basis_global = dut_rotation_matrix.T.dot(np.eye(3, dtype=np.float64))
                    dut_normal = basis_global[2]
                    dut_normal /= np.sqrt(np.dot(dut_normal, dut_normal))
                    if dut_normal[2] < 0:
                        dut_normal = -dut_normal
                    normal.append(dut_normal)
                x_global = np.column_stack(x_global)
                y_global = np.column_stack(y_global)
                z_global = np.column_stack(z_global)
                translation = np.row_stack(translation)
                normal = np.row_stack(normal)
                hit_flag = np.zeros_like(tracklets_data_chunk['hit_flag'])
                # Perform the track finding with jitted loop
                _find_tracks_loop(
                    event_numbers=event_numbers,
                    indices=indices,
                    z_sorted_dut_indices=z_sorted_dut_indices,
                    x=x_global,
                    y=y_global,
                    z=z_global,
                    select_extrapolation_duts=select_extrapolation_duts,
                    translation=translation,
                    normal=normal)
                # copy the columns to the result array
                for dut_index in range(n_duts):
                    for column_name in tracklets_data_chunk.dtype.names:
                        if 'dut_%d' % dut_index in column_name:
                            tracklets_data_chunk[column_name] = tracklets_data_chunk[column_name][indices[:, dut_index]]
                # calculate new hit flags
                for dut_index in range(n_duts):
                    hit_flag += np.isfinite(tracklets_data_chunk['x_dut_%d' % dut_index]).astype(hit_flag.dtype) << dut_index
                tracklets_data_chunk['hit_flag'] = hit_flag
                # append data to table
                track_candidates.append(tracklets_data_chunk)
                track_candidates.flush()
                total_n_events_stored_last = total_n_events_stored
                total_n_tracks_last = total_n_tracks
                last_index_chunk = index_chunk
                progress_bar.update(index_chunk)

            progress_bar.close()

    return output_track_candidates_file


@njit
def _find_tracks_loop(event_numbers, indices, z_sorted_dut_indices, x, y, z, select_extrapolation_duts, translation, normal):
    ''' This function provides an algorithm to generates the track candidates from the tracklets array.
    Each hit is put to the best fitting track. Tracks are assumed to have
    no big angle, otherwise this approach does not work.
    '''
    analyze_event_number = -1
    track_index = 0
    while track_index < event_numbers.shape[0]:
        curr_event_number = event_numbers[track_index]
        if curr_event_number != analyze_event_number or track_index == 0:  # Detect new event
            analyze_event_number = curr_event_number
            for dut_index in z_sorted_dut_indices[1:]:  # loop over all DUTs in the actual track
                track_index2 = track_index
                while track_index2 < event_numbers.shape[0]:
                    if analyze_event_number != event_numbers[track_index2]:
                        break
                    else:
                        _find_tracks(
                            event_numbers=event_numbers,
                            indices=indices,
                            z_sorted_dut_indices=z_sorted_dut_indices,
                            event_start_index=track_index,
                            track_index=track_index2,
                            dut_index=dut_index,
                            x=x,
                            y=y,
                            z=z,
                            select_extrapolation_duts=select_extrapolation_duts,
                            translation=translation,
                            normal=normal)
                    track_index2 += 1
        # goto next possible track
        track_index += 1


@njit
def _find_tracks(event_numbers, indices, z_sorted_dut_indices, event_start_index, track_index, dut_index, x, y, z, select_extrapolation_duts, translation, normal):
    swap = False
    # The hit distance of the actual assigned hit; -1 means not assigned
    first_reference_dut_index = _get_first_dut_index(
        x=x,
        track_index=track_index,
        z_sorted_dut_indices=z_sorted_dut_indices)
    sorted_dut_index = -1
    for i, val in enumerate(z_sorted_dut_indices):
        if val == dut_index:
            sorted_dut_index = i
    sorted_ref_dut_index = -1
    for i, val in enumerate(z_sorted_dut_indices):
        if val == first_reference_dut_index:
            sorted_ref_dut_index = i
    if sorted_ref_dut_index >= sorted_dut_index or first_reference_dut_index == -1:
        return

    curr_dut_index = dut_index
    reference_dut_index = -1
    cnt = 0
    while cnt < 2:
        sorted_dut_index = -1
        for i, val in enumerate(z_sorted_dut_indices):
            if val == curr_dut_index:
                sorted_dut_index = i
        if sorted_dut_index == 0:
            break
        reference_dut_index = _get_last_dut_index(
            x=x,
            track_index=track_index,
            z_sorted_dut_indices=z_sorted_dut_indices[:sorted_dut_index])
        if reference_dut_index == -1:
            break
        if cnt == 0 and reference_dut_index in select_extrapolation_duts:
            second_ref_dut_index = reference_dut_index
            cnt = cnt + 1
        elif cnt == 1 and reference_dut_index in select_extrapolation_duts:
            first_ref_dut_index = reference_dut_index
            cnt = cnt + 1
        curr_dut_index = reference_dut_index
    if cnt == 2:
        u_0 = x[track_index][second_ref_dut_index] - x[track_index][first_ref_dut_index]
        u_1 = y[track_index][second_ref_dut_index] - y[track_index][first_ref_dut_index]
        u_2 = z[track_index][second_ref_dut_index] - z[track_index][first_ref_dut_index]
        u_len = np.sqrt(u_0**2 + u_1**2 + u_2**2)
        u_0 /= u_len
        u_1 /= u_len
        u_2 /= u_len
        w_0 = x[track_index][second_ref_dut_index] - translation[dut_index][0]
        w_1 = y[track_index][second_ref_dut_index] - translation[dut_index][1]
        w_2 = z[track_index][second_ref_dut_index] - translation[dut_index][2]
        s_i = -(normal[dut_index][0] * w_0 + normal[dut_index][1] * w_1 + normal[dut_index][2] * w_2) / (normal[dut_index][0] * u_0 + normal[dut_index][1] * u_1 + normal[dut_index][2] * u_2)
        actual_reference_x = translation[dut_index][0] + w_0 + s_i * u_0
        actual_reference_y = translation[dut_index][1] + w_1 + s_i * u_1
    else:
        actual_reference_x, actual_reference_y = x[track_index][first_reference_dut_index], y[track_index][first_reference_dut_index]

    best_track_index = track_index
    if np.isnan(x[track_index][dut_index]):
        best_hit_distance = -1  # Value for no hit
    else:
        # Calculate the hit distance of the actual assigned DUT hit towards the actual reference hit
        best_hit_distance = math.sqrt((x[track_index][dut_index] - actual_reference_x)**2 + (y[track_index][dut_index] - actual_reference_y)**2)
    # The shortest hit distance to the actual hit; -1 means not assigned
    # for hit_index in range(actual_event_start_index, event_numbers.shape[0]):  # Loop over all not sorted hits of actual DUT
    hit_index = event_start_index
    while hit_index < event_numbers.shape[0]:
        if event_numbers[hit_index] != event_numbers[event_start_index]:  # Abort condition
            break
        if track_index == hit_index:  # Check if hit swapping is needed
            hit_index += 1
            continue
        actual_hit_x, actual_hit_y = x[hit_index][dut_index], y[hit_index][dut_index]
        if np.isnan(actual_hit_x):  # x = nan is no hit
            hit_index += 1
            continue
        # Calculate the hit distance of the actual DUT hit towards the actual reference hit
        actual_x_distance, actual_y_distance = actual_hit_x - actual_reference_x, actual_hit_y - actual_reference_y
        actual_hit_distance = math.sqrt(actual_x_distance**2 + actual_y_distance**2)
        if best_hit_distance >= 0 and best_hit_distance < actual_hit_distance:  # Check if actual assigned hit is better
            hit_index += 1
            continue
        # TODO: do not take all hits, check for valid hits (i.e., inside scatter cone)
        # Get reference DUT index of other track
        first_other_dut_hit_index = _get_first_dut_index(
            x=x,
            track_index=hit_index,
            z_sorted_dut_indices=z_sorted_dut_indices)
        if first_other_dut_hit_index != dut_index:
            curr_other_dut_index = dut_index
            other_reference_dut_index = -1
            cnt = 0
            while cnt < 2:
                sorted_dut_index = -1
                for i, val in enumerate(z_sorted_dut_indices):
                    if val == curr_other_dut_index:
                        sorted_dut_index = i
                if sorted_dut_index == 0:
                    break
                other_reference_dut_index = _get_last_dut_index(
                    x=x,
                    track_index=hit_index,
                    z_sorted_dut_indices=z_sorted_dut_indices[:sorted_dut_index])
                if other_reference_dut_index == -1:
                    break
                if cnt == 0 and other_reference_dut_index in select_extrapolation_duts:
                    second_other_ref_dut_index = other_reference_dut_index
                    cnt = cnt + 1
                elif cnt == 1 and other_reference_dut_index in select_extrapolation_duts:
                    first_other_ref_dut_index = other_reference_dut_index
                    cnt = cnt + 1
                curr_other_dut_index = other_reference_dut_index
            if cnt == 2:
                u_0 = x[hit_index][second_other_ref_dut_index] - x[hit_index][first_other_ref_dut_index]
                u_1 = y[hit_index][second_other_ref_dut_index] - y[hit_index][first_other_ref_dut_index]
                u_2 = z[hit_index][second_other_ref_dut_index] - z[hit_index][first_other_ref_dut_index]
                u_len = np.sqrt(u_0**2 + u_1**2 + u_2**2)
                u_0 /= u_len
                u_1 /= u_len
                u_2 /= u_len
                w_0 = x[hit_index][second_other_ref_dut_index] - translation[dut_index][0]
                w_1 = y[hit_index][second_other_ref_dut_index] - translation[dut_index][1]
                w_2 = z[hit_index][second_other_ref_dut_index] - translation[dut_index][2]
                s_i = -(normal[dut_index][0] * w_0 + normal[dut_index][1] * w_1 + normal[dut_index][2] * w_2) / (normal[dut_index][0] * u_0 + normal[dut_index][1] * u_1 + normal[dut_index][2] * u_2)
                reference_x_other = translation[dut_index][0] + w_0 + s_i * u_0
                reference_y_other = translation[dut_index][1] + w_1 + s_i * u_1
            else:
                reference_x_other, reference_y_other = x[hit_index][first_other_dut_hit_index], y[hit_index][first_other_dut_hit_index]
            # Calculate hit distance to reference hit of other track
            x_distance_other, y_distance_other = actual_hit_x - reference_x_other, actual_hit_y - reference_y_other
            hit_distance_other = math.sqrt(x_distance_other**2 + y_distance_other**2)
            if actual_hit_distance > hit_distance_other:  # Only take hit if it fits better to actual track; otherwise leave it with other track
                hit_index += 1
                continue
        # setting best hit
        best_track_index = hit_index
        best_hit_distance = actual_hit_distance
        hit_index += 1
        swap = True
    if not swap:
        return
    # swapping hits
    tmp_x, tmp_y, tmp_z = x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index]
    tmp_index = indices[track_index][dut_index]

    x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index] = x[best_track_index][dut_index], y[best_track_index][dut_index], z[best_track_index][dut_index]
    indices[track_index][dut_index] = indices[best_track_index][dut_index]

    x[best_track_index][dut_index], y[best_track_index][dut_index], z[best_track_index][dut_index] = tmp_x, tmp_y, tmp_z
    indices[best_track_index][dut_index] = tmp_index
    # recursively call _find_tracks in case of swapping
    # hits with other finished tracks
    first_dut_hit_index = _get_first_dut_index(
        x=x,
        track_index=best_track_index,
        z_sorted_dut_indices=z_sorted_dut_indices)
    if track_index > best_track_index and first_dut_hit_index != dut_index:
        _find_tracks(
            event_numbers=event_numbers,
            indices=indices,
            z_sorted_dut_indices=z_sorted_dut_indices,
            event_start_index=event_start_index,
            track_index=best_track_index,
            dut_index=dut_index,
            x=x,
            y=y,
            z=z,
            select_extrapolation_duts=select_extrapolation_duts,
            translation=translation,
            normal=normal)


@njit
def _get_first_dut_index(x, track_index, z_sorted_dut_indices):
    ''' Returns the first DUT that has a hit for the track at given index '''
    for dut_index in z_sorted_dut_indices:  # Loop over duts, to get first DUT hit of track
        if not np.isnan(x[track_index][dut_index]):
            return dut_index
    return -1


@njit
def _get_last_dut_index(x, track_index, z_sorted_dut_indices):
    ''' Returns the first DUT that has a hit for the track at given index '''
    for dut_index in z_sorted_dut_indices[::-1]:  # Loop over duts, to get first DUT hit of track
        if not np.isnan(x[track_index][dut_index]):
            return dut_index
    return -1


def fit_tracks(telescope_configuration, input_track_candidates_file, output_tracks_file=None, max_events=None, select_duts=None, select_hit_duts=None, select_fit_duts=None, min_track_hits=None, exclude_dut_hit=False, select_align_duts=None, method='fit', beam_energy=None, particle_mass=None, scattering_planes=None, quality_distances=(250.0, 250.0), isolation_distances=(500.0, 500.0), use_limits=True, keep_data=False, full_track_info=False, plot=True, chunk_size=1000000):
    '''Calculate tracks and set tracks quality flag for selected DUTs.
    Two methods are available to generate tracks: a linear fit (method="fit") and a Kalman Filter (method="kalman").

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_track_candidates_file : string
        Filename of the input track candidate file.
    output_tracks_file : string
        Filename of the output tracks file.
    max_events : uint
        Maximum number of randomly chosen events. If None, all events are taken.
    select_duts : list
        Specify the fit DUTs for which tracks will be fitted and a track array will be generated.
        If None, for all DUTs are selected.
    select_hit_duts : list or list of lists
        Specifying DUTs that are required to have a hit for each selected DUT.
        If None, no DUT is required to have a hit.
    select_fit_duts : list or list of lists
        Specifying DUTs that are used for the track fit for each selected DUT.
        If None, all DUTs are used for the track fit.
        Note: This parameter needs to be set correctly. Usually not all available DUTs should be used for track fitting.
        The list usually only contains DUTs, which are part of the telescope.
    min_track_hits : uint or list
        Minimum number of track hits for each selected DUT.
        If None or list item is None, the minimum number of track hits is the length of select_fit_duts.
    exclude_dut_hit : bool or list
        Decide whether or not to use hits in the actual fit DUT for track fitting (for unconstrained residuals).
        If False (default), use all DUTs as specified in select_fit_duts and use them for track fitting if hits are available (potentially constrained residuals).
        If True, do not use hits form the actual fit DUT for track fitting, even if specified in select_fit_duts (unconstrained residuals).
    select_align_duts : list
        Specify the DUTs for which a residual offset correction is to be carried out.
        Note: This functionality is only used for the alignment of the DUTs.
    method : string
        Available methods are 'kalman', which uses a Kalman Filter for track calculation, and 'fit', which uses a simple
        straight line fit for track calculation.
    beam_energy : float
        Energy of the beam in MeV, e.g., 2500.0 MeV for ELSA beam. Only used for the Kalman Filter.
    particle_mass : float
        Mass of the particle in MeV, e.g., 0.511 MeV for electrons. Only used for the Kalman Filter.
    scattering_planes : list or dict
        Specifies additional scattering planes in case of DUTs which are not used or additional material in the way of the tracks.
        The list must contain dictionaries containing the following keys:
            material_budget: material budget of the scattering plane
            translation_x/translation_y/translation_z: x/y/z position of the plane (in um)
            rotation_alpha/rotation_beta/rotation_gamma: alpha/beta/gamma angle of scattering plane (in radians)
        The material budget is defined as the thickness devided by the radiation length.
        If scattering_planes is None, no scattering plane will be added.
    quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. The selected track and corresponding hit
        must have a smaller distance to have the quality flag to be set to 1.
        If None, set distance to infinite.
    isolation_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the isolated track/hit flag. Any other occurence of tracks or hits from the same event
        within this distance will prevent the flag from beeing set.
        If None, set distance to 0.
    use_limits : bool
        If True, use column and row limits from pre-alignment for selecting the data.
    keep_data : bool
        Keep all track candidates in data and add track info only to fitted tracks. Necessary for purity calculations.
    full_track_info : bool
        If True, the track vector and position of all DUTs is appended to track table in order to get the full track information.
        If False, only the track vector and position of the actual fit DUT is appended to track table.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_tracks_file : string
        Filename of the output tracks file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Fitting tracks of %d DUTs ===' % n_duts)

    method = method.lower()
    if method not in ["fit", "kalman"]:
        raise ValueError('Unknown method "%s"!' % method)

    if method == "kalman" and not beam_energy:
        raise ValueError('Beam energy not given (in MeV).')

    if method == "kalman" and not particle_mass:
        raise ValueError('Particle mass not given (in MeV).')

    if method == "fit":
        method_str = 'Straight Line Fit'
    if method == "kalman":
        method_str = 'Kalman Filter'
    logging.info('=== Fitting tracks of %d DUTs (using %s)===' % (n_duts, method_str))

    if output_tracks_file is None:
        output_tracks_file = os.path.join(os.path.dirname(input_track_candidates_file), 'Tracks_%s.h5' % method.title())
    if plot:
        output_pdf_file = os.path.splitext(output_tracks_file)[0] + '.pdf'
        output_pdf = PdfPages(output_pdf_file, keep_empty=False)
    else:
        output_pdf = None

    if select_duts is None:
        select_duts = list(range(n_duts))  # standard setting: fit tracks for all DUTs
    elif not isinstance(select_duts, Iterable):
        select_duts = [select_duts]
    # Check for duplicates
    if len(select_duts) != len(set(select_duts)):
        raise ValueError("found douplicate in select_duts")
    # Check if any iterable in iterable
    if any(map(lambda val: isinstance(val, Iterable), select_duts)):
        raise ValueError("item in select_duts is iterable")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = list(range(n_duts))
        # # copy each item
        # for hit_duts in select_hit_duts:
        #     select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("select_fit_duts is no iterable")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("select_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in select_duts]
    # if None use all DUTs
    for index, item in enumerate(select_fit_duts):
        if item is None:
            select_fit_duts[index] = list(range(n_duts))
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("not all items in select_fit_duts are iterable")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_fit_duts has the wrong length")
    for index, item in enumerate(select_fit_duts):
        if len(item) < 2:  # check the length of the items
            raise ValueError("item in select_fit_duts has length < 2")

    # Create track, hit selection
    if select_hit_duts is None:  # If None, require no hit
        # select_hit_duts = list(range(n_duts))
        select_hit_duts = []
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("select_hit_duts is no iterable")
    # elif not select_hit_duts:  # empty iterable
    #     raise ValueError("select_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # If None, require no hit
    for index, item in enumerate(select_hit_duts):
        if item is None:
            select_hit_duts[index] = []
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("not all items in select_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_hit_duts has the wrong length")

    # Create quality distance
    if isinstance(quality_distances, tuple) or quality_distances is None:
        quality_distances = [quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(quality_distances, Iterable):
        raise ValueError("quality_distances is no iterable")
    elif not quality_distances:  # empty iterable
        raise ValueError("quality_distances has no items")
    # Finally check length of all arrays
    if len(quality_distances) != n_duts:  # empty iterable
        raise ValueError("quality_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, quality_distances)):
        raise ValueError("not all items in quality_distances are iterable")
    # Finally check length of all arrays
    for distance in quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in quality_distances has length != 2")

    # Create reject quality distance
    if isinstance(isolation_distances, tuple) or isolation_distances is None:
        isolation_distances = [isolation_distances] * n_duts
    # Check iterable and length
    if not isinstance(isolation_distances, Iterable):
        raise ValueError("isolation_distances is no iterable")
    elif not isolation_distances:  # empty iterable
        raise ValueError("isolation_distances has no items")
    # Finally check length of all arrays
    if len(isolation_distances) != n_duts:  # empty iterable
        raise ValueError("isolation_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, isolation_distances)):
        raise ValueError("not all items in isolation_distances are iterable")
    # Finally check length of all arrays
    for distance in isolation_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in isolation_distances has length != 2")

    # Check iterable and length
    if not isinstance(exclude_dut_hit, Iterable):
        exclude_dut_hit = [exclude_dut_hit] * len(select_duts)
    elif not exclude_dut_hit:  # empty iterable
        raise ValueError("exclude_dut_hit has no items")
    # Finally check length of all array
    if len(exclude_dut_hit) != len(select_duts):  # empty iterable
        raise ValueError("exclude_dut_hit has the wrong length")
    # Check if only bools in iterable
    if not all(map(lambda val: isinstance(val, (bool,)), exclude_dut_hit)):
        raise ValueError("not all items in exclude_dut_hit are boolean")

    # Check iterable and length
    if not isinstance(min_track_hits, Iterable):
        min_track_hits = [min_track_hits] * len(select_duts)
    # Finally check length of all arrays
    if len(min_track_hits) != len(select_duts):  # empty iterable
        raise ValueError("min_track_hits has the wrong length")

    fitted_duts = []
    pool = Pool()
    with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
        with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
            for fit_dut_index, actual_fit_dut in enumerate(select_duts):  # Loop over the DUTs where tracks shall be fitted for
                # Test whether other DUTs have identical tracks
                # if yes, save some CPU time and fit only once.
                # This following list contains all DUT indices that will be fitted
                # during this step of the loop.
                if actual_fit_dut in fitted_duts:
                    continue
                # calculate all DUTs with identical tracks to save processing time
                actual_fit_duts = []
                for curr_fit_dut_index, curr_fit_dut in enumerate(select_duts):
                    if (curr_fit_dut == actual_fit_dut or
                        (((exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is False and set(select_fit_duts[curr_fit_dut_index]) == set(select_fit_duts[fit_dut_index])) or
                          (exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is True and set(select_fit_duts[curr_fit_dut_index]) == (set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut]))) or
                          (exclude_dut_hit[curr_fit_dut_index] is True and exclude_dut_hit[fit_dut_index] is False and (set(select_fit_duts[curr_fit_dut_index]) - set([curr_fit_dut])) == set(select_fit_duts[fit_dut_index])) or
                          (exclude_dut_hit[curr_fit_dut_index] is True and exclude_dut_hit[fit_dut_index] is True and (set(select_fit_duts[curr_fit_dut_index]) - set([curr_fit_dut])) == (set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut])))) and
                         set(select_hit_duts[curr_fit_dut_index]) == set(select_hit_duts[fit_dut_index]) and
                         min_track_hits[curr_fit_dut_index] == min_track_hits[fit_dut_index])):
                        actual_fit_duts.append(curr_fit_dut)
                # continue with fitting
                logging.info('= Fit tracks for %s =', ', '.join([telescope[curr_dut].name for curr_dut in actual_fit_duts]))
                # remove existing nodes
                for dut_index in actual_fit_duts:
                    try:  # Check if table already exists, then append data
                        out_file_h5.remove_node(out_file_h5.root, name='Tracks_DUT%d' % dut_index)
                        logging.info('Overwriting existing tracks from %s', telescope[curr_dut].name)
                    except tb.NodeError:  # Table does not exist, thus create new
                        pass

                total_n_tracks = in_file_h5.root.TrackCandidates.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0

                # select hit DUTs based on input parameters
                # hit DUTs are always enforced
                hit_duts = select_hit_duts[fit_dut_index]
                dut_hit_mask = 0  # DUTs required to have hits
                for dut_index in hit_duts:
                    dut_hit_mask |= ((1 << dut_index))
                logging.info('Require hits in %d DUTs for track selection: %s', len(hit_duts), ', '.join([telescope[curr_dut].name for curr_dut in hit_duts]))

                # select fit DUTs based on input parameters
                # exclude actual DUTs from fit DUTs if exclude_dut_hit parameter is set (for, e.g., unbiased residuals)
                fit_duts = list(set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut])) if exclude_dut_hit[fit_dut_index] else select_fit_duts[fit_dut_index]
                if min_track_hits[fit_dut_index] is None:
                    actual_min_track_hits = len(fit_duts)
                else:
                    actual_min_track_hits = min_track_hits[fit_dut_index]
                if actual_min_track_hits < 2:
                    raise ValueError('The number of required hits is smaller than 2. Cannot fit tracks for %s.', telescope[actual_fit_dut].name)
                dut_fit_mask = 0  # DUTs to be used for the fit
                for dut_index in fit_duts:
                    dut_fit_mask |= ((1 << dut_index))
                if actual_min_track_hits > len(fit_duts):
                    raise RuntimeError("min_track_hits for DUT%d is larger than the number of fit DUTs" % (actual_fit_dut,))
                logging.info('Require at least %d hits in %d DUTs for track selection: %s', actual_min_track_hits, len(fit_duts), ', '.join([telescope[curr_dut].name for curr_dut in fit_duts]))

                # selecting DUTs for residual correction
                if select_align_duts is not None and select_align_duts:
                    logging.info("Correct residual offset for %d DUTs: %s", len(select_align_duts), ', '.join([telescope[curr_dut].name for curr_dut in select_align_duts]))

                progress_bar = tqdm(total=total_n_tracks, ncols=80)
                # progress_bar = tqdm(total=max_tracks if max_tracks is not None else in_file_h5.root.TrackCandidates.shape[0], ncols=80)

                chunk_indices = []
                chunk_stats = []
                dut_stats = []
                for track_candidates_chunk, index_chunk in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=chunk_size):
                    chunk_indices.append(index_chunk)
                    # if max_tracks is not None and total_n_tracks >= max_tracks:
                    #     break
                    n_tracks_chunk = track_candidates_chunk.shape[0]
                    # selecting data with 2 or more hits in the fit DUTs for fitting;
                    # this rquirement will have an impact on isolated_track_flag;
                    # the more tracks are fitted, the calculation of isolated_track_flag becomes better.
                    # NOTE: this value has a significant impact on CPU processing time
                    if method == "kalman":
                        # increase minimum track hits requirement for Kalman fit to reduce CPU processing time
                        select_fit_tracks = (analysis_utils.number_of_set_bits(track_candidates_chunk['hit_flag'] & dut_fit_mask) >= actual_min_track_hits)
                        # select_fit_tracks = (analysis_utils.number_of_set_bits(track_candidates_chunk['hit_flag'] & dut_fit_mask) >= min(3, actual_min_track_hits))
                    else:
                        select_fit_tracks = (analysis_utils.number_of_set_bits(track_candidates_chunk['hit_flag'] & dut_fit_mask) >= 2)
                    # Select tracks that will be stored
                    select_tracks_for_storage = select_fit_tracks.copy()
                    # ... select tracks with DUTs that are required to have a hit
                    select_tracks_for_storage &= ((track_candidates_chunk['hit_flag'] & dut_hit_mask) == dut_hit_mask)
                    # ... and fulfilling the minimum track hits requirement
                    select_tracks_for_storage &= (analysis_utils.number_of_set_bits(track_candidates_chunk['hit_flag'] & dut_fit_mask) >= actual_min_track_hits)
                    n_good_tracks = np.count_nonzero(select_tracks_for_storage)
                    chunk_stats.append(n_good_tracks / n_tracks_chunk)
                    # remove tracks that have only a single DUT with a hit
                    # for index, bit in enumerate(np.binary_repr(dut_fit_mask)[::-1]):  # iterate from LSB to MSB
                    #     if bit == "0":
                    #         continue
                    #     dut_fit_mask_dut_removed = dut_fit_mask & ~(1 << index)
                    #     select_fit_tracks &= (track_candidates_chunk['hit_flag'] & dut_fit_mask_dut_removed) > 0

                    # if max_tracks is not None:
                    #     cut_index = np.where(np.cumsum(select_fit_tracks) + total_n_tracks > max_tracks)[0]
                    #     print "cut index", cut_index
                    #     if len(cut_index) > 0:
                    #         event_indices = np.where(track_candidates_chunk["event_number"][:-1] != track_candidates_chunk["event_number"][1:])[0] + 1
                    #         event_cut_index = event_indices[event_indices >= cut_index[0]][0]
                    #         # print track_candidates_chunk[event_cut_index-2:event_cut_index+2]["event_number"]
                    #         # print track_candidates_chunk[event_cut_index-2:event_cut_index]["event_number"]
                    #         select_fit_tracks = select_fit_tracks[:event_cut_index]
                    #         track_candidates_chunk = track_candidates_chunk[:event_cut_index]
                    #         # print "event_cut_index", event_cut_index, total_n_tracks, max_tracks

                    unique_events = np.unique(track_candidates_chunk["event_number"][select_tracks_for_storage])
                    n_events_chunk = unique_events.shape[0]
                    if n_events_chunk == 0:
                        continue

                    if max_events:
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                        else:
                            # calculate correction of number of selected events
                            correction = (total_n_tracks - index_chunk) / total_n_tracks * 1 / (((total_n_tracks - last_index_chunk) / total_n_tracks) / ((max_events - total_n_events_stored_last) / max_events)) \
                                + (index_chunk) / total_n_tracks * 1 / (((last_index_chunk) / total_n_tracks) / ((total_n_events_stored_last) / max_events))
                            # select_n_events = np.ceil(n_events_chunk * correction)
                            # calculate correction of number of selected events
                            # correction = 1/(((total_n_tracks-last_index_chunk)/total_n_tracks_last)/((max_events-total_n_events_stored_last)/max_events))
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                            # print "correction", correction
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(select_n_events, max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        # print "store_n_events", store_n_events
                        select_tracks_for_storage &= np.in1d(track_candidates_chunk["event_number"], selected_events)
                        # TODO: total_n_tracks_stored not used...
                        store_n_tracks = np.count_nonzero(select_tracks_for_storage)
                        total_n_tracks_stored += store_n_tracks
                        # track_candidates_chunk = track_candidates_chunk[select_tracks]

                    # Prepare track hits array to be fitted
                    n_fit_tracks = np.count_nonzero(select_fit_tracks)  # Index of tmp track hits array
                    if method == "fit":
                        track_hits = np.full((n_fit_tracks, len(fit_duts), 3), fill_value=np.nan, dtype=np.float64)
                    elif method == "kalman":
                        track_hits = np.full((n_fit_tracks, n_duts, 6), fill_value=np.nan, dtype=np.float64)

                    fit_array_index = 0
                    for dut_index, dut in enumerate(telescope):  # Fill index loop of new array
                        # Check if DUT is used for fit
                        if method == "fit" and dut_index in fit_duts:
                            # apply alignment for fitting the tracks
                            track_hits[:, fit_array_index, 0], track_hits[:, fit_array_index, 1], track_hits[:, fit_array_index, 2] = dut.local_to_global_position(
                                x=track_candidates_chunk['x_dut_%s' % dut_index][select_fit_tracks],
                                y=track_candidates_chunk['y_dut_%s' % dut_index][select_fit_tracks],
                                z=track_candidates_chunk['z_dut_%s' % dut_index][select_fit_tracks])
                            # increase index for tracks hits array
                            fit_array_index += 1
                        elif method == "kalman":
                            # TODO: taking telescope alignment into account for initial state
                            # apply alignment for fitting the tracks
                            track_hits[:, dut_index, 0], track_hits[:, dut_index, 1], track_hits[:, dut_index, 2] = dut.local_to_global_position(
                                x=track_candidates_chunk['x_dut_%s' % dut_index][select_fit_tracks],
                                y=track_candidates_chunk['y_dut_%s' % dut_index][select_fit_tracks],
                                z=track_candidates_chunk['z_dut_%s' % dut_index][select_fit_tracks])
                            track_hits[:, dut_index, 3], track_hits[:, dut_index, 4], track_hits[:, dut_index, 5] = np.abs(dut.local_to_global_position(
                                x=track_candidates_chunk['x_err_dut_%s' % dut_index][select_fit_tracks],
                                y=track_candidates_chunk['y_err_dut_%s' % dut_index][select_fit_tracks],
                                z=track_candidates_chunk['z_err_dut_%s' % dut_index][select_fit_tracks],
                                # no translation for the errors
                                translation_x=0.0,
                                translation_y=0.0,
                                translation_z=0.0))

                    # Split data and fit on all available cores
                    n_slices = cpu_count()
                    track_hits_slices = np.array_split(track_hits, n_slices)
                    if method == "fit":
                        results = [pool.apply_async(_fit_tracks_loop, kwds={
                            'track_hits': track_hits_slice}) for track_hits_slice in track_hits_slices]
                    elif method == "kalman":
                        results = [pool.apply_async(_fit_tracks_kalman_loop, kwds={
                            'track_hits': track_hits_slice,
                            'telescope': telescope,
                            'select_fit_duts': fit_duts,
                            'beam_energy': beam_energy,
                            'particle_mass': particle_mass,
                            'scattering_planes': scattering_planes}) for track_hits_slice in track_hits_slices]

                    # Store results
                    offsets = np.concatenate([result.get()[0] for result in results])  # Merge offsets from all cores in results
                    slopes = np.concatenate([result.get()[1] for result in results])  # Merge slopes from all cores in results
                    if method == 'kalman':
                        track_chi2s = np.concatenate([result.get()[2] for result in results])  # Merge track chi2 from all cores in results
                    else:
                        track_chi2s = None
                    # Store the data
                    # Check if all DUTs were fitted at once
                    dut_stats.append(store_track_data(
                        out_file_h5=out_file_h5,
                        track_candidates_chunk=track_candidates_chunk,
                        select_fit_tracks=select_fit_tracks,
                        select_tracks_for_storage=select_tracks_for_storage,
                        telescope=telescope,
                        offsets=offsets,
                        slopes=slopes,
                        track_chi2s=track_chi2s,
                        fit_duts=actual_fit_duts,  # storing tracks for these DUTs
                        select_fit_duts=fit_duts,  # DUTs used for fitting tracks
                        select_align_duts=select_align_duts,
                        quality_distances=quality_distances,
                        isolation_distances=isolation_distances,
                        use_limits=use_limits,
                        keep_data=keep_data,
                        method=method,
                        full_track_info=full_track_info))

                    # total_n_tracks += n_good_tracks
                    total_n_events_stored_last = total_n_events_stored
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
                    # progress_bar.update(min(total_n_tracks, max_tracks) if max_tracks is not None else index_chunk)
                progress_bar.close()
                # print "***************"
                # print "total_n_tracks_stored", total_n_tracks_stored
                # print "total_n_events_stored", total_n_events_stored
                fitted_duts.extend(actual_fit_duts)

                try:
                    plot_utils.plot_fit_tracks_statistics(
                        telescope=telescope,
                        fit_duts=actual_fit_duts,
                        chunk_indices=chunk_indices,
                        chunk_stats=chunk_stats,
                        dut_stats=dut_stats,
                        output_pdf=output_pdf)
                except Exception as e:
                    print(e)

    pool.close()
    pool.join()

    if output_pdf is not None:
        output_pdf.close()

    if plot:
        plot_utils.plot_track_chi2(input_tracks_file=output_tracks_file, output_pdf_file=None, dut_names=telescope.dut_names, chunk_size=chunk_size)

    return output_tracks_file


def store_track_data(out_file_h5, track_candidates_chunk, select_fit_tracks, select_tracks_for_storage, telescope, offsets, slopes, track_chi2s, fit_duts, select_fit_duts, select_align_duts, quality_distances, isolation_distances, use_limits, keep_data, method, full_track_info):
    dut_stats = []
    fit_duts_offsets = []
    fit_duts_slopes = []
    n_fit_tracks = np.count_nonzero(select_fit_tracks)
    n_good_tracks = np.count_nonzero(select_tracks_for_storage)
    # xy_residuals_squared = np.empty((n_good_tracks, len(telescope)), dtype=np.float64)
    x_residuals_squared = np.empty((n_fit_tracks, len(telescope)), dtype=np.float64)
    x_err_squared = np.empty((n_fit_tracks, len(telescope)), dtype=np.float64)
    y_residuals_squared = np.empty((n_fit_tracks, len(telescope)), dtype=np.float64)
    y_err_squared = np.empty((n_fit_tracks, len(telescope)), dtype=np.float64)
    # reset quality and isolation flag
    quality_flag = np.zeros(n_fit_tracks, dtype=track_candidates_chunk["hit_flag"].dtype)
    isolated_hit_flag = np.zeros_like(quality_flag)
    isolated_track_flag = np.zeros_like(quality_flag)
    if full_track_info:
        track_estimates_chunk_full = np.full(shape=(n_fit_tracks, len(telescope), 6), fill_value=np.nan, dtype=np.float64)
    else:
        track_estimates_chunk_full = None
    for dut_index, dut in enumerate(telescope):
        dut_stats.append([])
        if use_limits:
            limit_x_local = dut.x_limit  # (lower limit, upper limit)
            limit_y_local = dut.y_limit  # (lower limit, upper limit)
        else:
            limit_x_local = None
            limit_y_local = None

        hit_x_local = track_candidates_chunk['x_dut_%s' % dut_index][select_fit_tracks]
        hit_y_local = track_candidates_chunk['y_dut_%s' % dut_index][select_fit_tracks]

        if method == "fit":
            # Set the offset to the track intersection with the tilted plane
            intersections_global = geometry_utils.get_line_intersections_with_dut(
                line_origins=offsets,
                line_directions=slopes,
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
            slopes_x_local, slopes_y_local, slopes_z_local = dut.global_to_local_position(
                x=slopes[:, 0],
                y=slopes[:, 1],
                z=slopes[:, 2],
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
        elif method in ["kalman", "ckf"]:
            # Set the offset to the track intersection with the tilted plane
            intersections_global = geometry_utils.get_line_intersections_with_dut(
                line_origins=offsets[:, dut_index],
                line_directions=slopes[:, dut_index],
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
            slopes_x_local, slopes_y_local, slopes_z_local = dut.global_to_local_position(
                x=slopes[:, dut_index, 0],
                y=slopes[:, dut_index, 1],
                z=slopes[:, dut_index, 2],
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

        # force the 3rd component (z) to be positive
        # and normalize to 1
        slopes_local = np.column_stack([slopes_x_local, slopes_y_local, slopes_z_local])
        slopes_local[slopes_local[:, 2] < 0.0] = -slopes_local[slopes_local[:, 2] < 0.0]
        slope_local_mag = np.sqrt(np.einsum('ij,ij->i', slopes_local, slopes_local))
        slopes_local /= slope_local_mag[:, np.newaxis]
        slopes_x_local, slopes_y_local, slopes_z_local = slopes_local[:, 0], slopes_local[:, 1], slopes_local[:, 2]

        intersection_x_local, intersection_y_local, intersection_z_local = dut.global_to_local_position(
            x=intersections_global[:, 0],
            y=intersections_global[:, 1],
            z=intersections_global[:, 2])

        x_residuals = hit_x_local - intersection_x_local
        y_residuals = hit_y_local - intersection_y_local
        select_finite_distance = np.isfinite(x_residuals)
        select_finite_distance &= np.isfinite(y_residuals)

        select_fitted_tracks_for_storage = select_tracks_for_storage[select_fit_tracks]
        n_good_tracks_with_hits = np.count_nonzero(select_fitted_tracks_for_storage & select_finite_distance)
        if n_good_tracks == 0:
            dut_stats[dut_index].append(0)
        else:
            dut_stats[dut_index].append(n_good_tracks_with_hits / n_good_tracks)

        # correction for residuals if residual mean (peak) is far away from 0
        if select_align_duts is not None and dut_index in select_align_duts and n_good_tracks_with_hits > 10:  # check for size of array, usually the last chunk affected
            center_x = np.median(x_residuals[select_fitted_tracks_for_storage & select_finite_distance])
            std_x = np.std(x_residuals[select_fitted_tracks_for_storage & select_finite_distance])
            x_kde = gaussian_kde(x_residuals[select_fitted_tracks_for_storage & select_finite_distance], bw_method=dut.pixel_size[0] / 12**0.5 / std_x)
            x_grid = np.linspace(center_x - 0.5 * std_x, center_x + 0.5 * std_x, np.ceil(std_x))
            x_index_max = np.argmax(x_kde.evaluate(x_grid))
            mean_x_residual = x_grid[x_index_max]
            x_residuals -= mean_x_residual
            center_y = np.median(y_residuals[select_fitted_tracks_for_storage & select_finite_distance])
            std_y = np.std(y_residuals[select_fitted_tracks_for_storage & select_finite_distance])
            y_kde = gaussian_kde(y_residuals[select_fitted_tracks_for_storage & select_finite_distance], bw_method=dut.pixel_size[1] / 12**0.5 / std_y)
            y_grid = np.linspace(center_y - 0.5 * std_y, center_y + 0.5 * std_y, np.ceil(std_y))
            y_index_max = np.argmax(y_kde.evaluate(y_grid))
            mean_y_residual = y_grid[y_index_max]
            y_residuals -= mean_y_residual

        # xy_residuals_squared[:, dut_index] = np.square(hit_x_local - intersection_x_local) + np.square(hit_y_local - intersection_y_local)
        x_residuals_squared[:, dut_index] = np.square(x_residuals)
        x_err_squared[:, dut_index] = np.square(track_candidates_chunk['x_err_dut_%d' % dut_index][select_fit_tracks])
        y_residuals_squared[:, dut_index] = np.square(y_residuals)
        y_err_squared[:, dut_index] = np.square(track_candidates_chunk['y_err_dut_%d' % dut_index][select_fit_tracks])

        # generate quality array
        dut_quality_flag_sel = np.ones_like(intersection_x_local, dtype=np.bool)
        dut_quality_flag_sel[~select_finite_distance] = False
        # select tracks within limits and set quality flag
        if limit_x_local is not None and np.isfinite(limit_x_local[0]):
            dut_quality_flag_sel &= (hit_x_local >= limit_x_local[0])
        if limit_x_local is not None and np.isfinite(limit_x_local[1]):
            dut_quality_flag_sel &= (hit_x_local <= limit_x_local[1])
        if limit_y_local is not None and np.isfinite(limit_y_local[0]):
            dut_quality_flag_sel &= (hit_y_local >= limit_y_local[0])
        if limit_y_local is not None and np.isfinite(limit_y_local[1]):
            dut_quality_flag_sel &= (hit_y_local <= limit_y_local[1])

        # distance for quality flag calculation
        if quality_distances[dut_index] is None:
            quality_distance_x = np.inf
            quality_distance_y = np.inf
        else:
            quality_distance_x = quality_distances[dut_index][0]
            quality_distance_y = quality_distances[dut_index][1]

        if quality_distance_x < 0 or quality_distance_y < 0:
            raise ValueError("Must use non-negative values for quality distance.")

        # select data where distance between the hit and track is smaller than the given value and set quality flag
        if quality_distance_x >= 2.5 * dut.pixel_size[0] and quality_distance_y >= 2.5 * dut.pixel_size[1]:  # use ellipse
            use_ellipse = True
            dut_quality_flag_sel[select_finite_distance] &= ((x_residuals_squared[select_finite_distance, dut_index] / quality_distance_x**2) + (y_residuals_squared[select_finite_distance, dut_index] / quality_distance_y**2)) <= 1
        else:  # use square
            use_ellipse = False
            dut_quality_flag_sel[select_finite_distance] &= (np.abs(x_residuals[select_finite_distance]) <= quality_distance_x)
            dut_quality_flag_sel[select_finite_distance] &= (np.abs(y_residuals[select_finite_distance]) <= quality_distance_y)
        quality_flag[dut_quality_flag_sel] |= np.uint32((1 << dut_index))

        n_good_tracks_with_hits_with_quality = np.count_nonzero(select_fitted_tracks_for_storage & select_finite_distance & dut_quality_flag_sel)
        if n_good_tracks_with_hits == 0:
            dut_stats[dut_index].append(0)
        else:
            dut_stats[dut_index].append(n_good_tracks_with_hits_with_quality / n_good_tracks_with_hits)

        # distance to find close-by hits and tracks
        if isolation_distances[dut_index] is None:
            isolation_distance_x = 0.0
            isolation_distance_y = 0.0
        else:
            isolation_distance_x = isolation_distances[dut_index][0]
            isolation_distance_y = isolation_distances[dut_index][1]

        if isolation_distance_x < 0 or isolation_distance_y < 0:
            raise ValueError("Must use non-negative values for reject quality distance.")

        # Select tracks that are too close when extrapolated to the actual DUT
        # All selected tracks will result in a quality_flag = 0 for the actual DUT
        dut_small_track_distance_flag_sel = np.zeros_like(dut_quality_flag_sel)
        _find_small_distance(
            event_number_array=track_candidates_chunk['event_number'][select_fit_tracks],
            position_array_x=intersection_x_local,
            position_array_y=intersection_y_local,
            max_distance_x=isolation_distance_x,
            max_distance_y=isolation_distance_y,
            small_distance_flag_array=dut_small_track_distance_flag_sel,
            use_ellipse=True)
        # Set flag for too close tracks (isolation flag is set to zero)
        isolated_track_flag[~dut_small_track_distance_flag_sel] |= np.uint32((1 << dut_index))

        n_good_tracks_with_hits_with_isolated_track = np.count_nonzero(select_fitted_tracks_for_storage & select_finite_distance & ~dut_small_track_distance_flag_sel)
        if n_good_tracks_with_hits == 0:
            dut_stats[dut_index].append(0)
        else:
            dut_stats[dut_index].append(n_good_tracks_with_hits_with_isolated_track / n_good_tracks_with_hits)

        # Select hits that are too close in a DUT
        # All selected hits will result in a quality_flag = 0 for the actual DUT
        dut_small_hit_distance_flag_sel = np.zeros_like(select_fit_tracks)
        _find_small_distance(
            event_number_array=track_candidates_chunk['event_number'],
            position_array_x=track_candidates_chunk['x_dut_%s' % dut_index],
            position_array_y=track_candidates_chunk['y_dut_%s' % dut_index],
            max_distance_x=isolation_distance_x,
            max_distance_y=isolation_distance_y,
            small_distance_flag_array=dut_small_hit_distance_flag_sel,
            use_ellipse=use_ellipse)
        # Set flag for too close hits (isolation flag is set to zero)
        isolated_hit_flag[~dut_small_hit_distance_flag_sel[select_fit_tracks]] |= np.uint32((1 << dut_index))

        n_good_tracks_with_hits_with_isolated_hit = np.count_nonzero(select_fitted_tracks_for_storage & select_finite_distance & ~dut_small_hit_distance_flag_sel[select_fit_tracks])
        if n_good_tracks_with_hits == 0:
            dut_stats[dut_index].append(0)
        else:
            dut_stats[dut_index].append(n_good_tracks_with_hits_with_isolated_hit / n_good_tracks_with_hits)

        n_isolated_and_quality_tracks = np.count_nonzero(dut_quality_flag_sel & ~dut_small_track_distance_flag_sel & ~dut_small_hit_distance_flag_sel[select_fit_tracks])
        dut_stats[dut_index].append(n_isolated_and_quality_tracks / track_candidates_chunk.shape[0])

        if dut_index in fit_duts:
            # use offsets at the location of the fit DUT, local coordinates
            fit_duts_offsets.append(np.column_stack([
                intersection_x_local,
                intersection_y_local,
                intersection_z_local]))
            # use slopes at the location of the fit DUT, local coordinates
            fit_duts_slopes.append(np.column_stack([
                slopes_x_local,
                slopes_y_local,
                slopes_z_local]))
        else:
            fit_duts_offsets.append(None)
            fit_duts_slopes.append(None)

        if full_track_info:
            track_estimates_chunk_full[:, dut_index] = np.column_stack([
                intersection_x_local,
                intersection_y_local,
                intersection_z_local,
                slopes_x_local,
                slopes_y_local,
                slopes_z_local])

    # Calculate chi2
    if method == 'fit':
        # calculate the sum of the squared x/y residuals of the fit DUT planes in the local coordinate system, divided by n fit DUT hits per track (normalization)
        # track_chi2s = np.sum(np.ma.masked_invalid(xy_residuals_squared[:, select_fit_duts]), axis=1) / np.count_nonzero(~np.isnan(xy_residuals_squared[:, select_fit_duts]), axis=1)
        track_chi2s = (np.sum(np.ma.masked_invalid(x_residuals_squared[:, select_fit_duts] / x_err_squared[:, select_fit_duts]), axis=1) + np.sum(np.ma.masked_invalid(y_residuals_squared[:, select_fit_duts] / y_err_squared[:, select_fit_duts]), axis=1))
        # select tracks that have more than 2 data points
        select_nonzero = (track_chi2s != 0.0)
        # divide by d.o.f.
        track_chi2s[select_nonzero] /= (2 * (np.count_nonzero(~np.isnan(x_residuals_squared[:, select_fit_duts]), axis=1)[select_nonzero] - 2))

    for dut_index, dut in enumerate(telescope):
        if dut_index not in fit_duts:
            continue
        tracks_array = create_results_array(
            n_duts=len(telescope),
            dut_offsets=fit_duts_offsets[dut_index],
            dut_slopes=fit_duts_slopes[dut_index],
            track_chi2s=track_chi2s,
            quality_flag=quality_flag,
            isolated_track_flag=isolated_track_flag,
            isolated_hit_flag=isolated_hit_flag,
            select_fit_tracks=select_fit_tracks,
            select_tracks_for_storage=select_tracks_for_storage,
            track_candidates_chunk=track_candidates_chunk,
            keep_data=keep_data,
            track_estimates_chunk_full=track_estimates_chunk_full)

        try:  # Check if table exists already, then append data
            tracklets_table = out_file_h5.get_node('/Tracks_DUT%d' % dut_index)
        except tb.NoSuchNodeError:  # Table does not exist, thus create new
            tracklets_table = out_file_h5.create_table(
                where=out_file_h5.root,
                name='Tracks_DUT%d' % dut_index,
                description=tracks_array.dtype,
                title='%s tracks for DUT%d' % (method.title(), dut_index),
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

        tracklets_table.append(tracks_array)
        tracklets_table.flush()

    return dut_stats


@njit
def _find_small_distance(event_number_array, position_array_x, position_array_y, max_distance_x, max_distance_y, small_distance_flag_array, use_ellipse):
    max_index = event_number_array.shape[0]
    index = 0
    while index < max_index:
        current_event_number = event_number_array[index]
        while (index < max_index) and (event_number_array[index] == current_event_number):  # Next event reached, break loop
            event_index = index + 1
            while (event_index < max_index) and (event_number_array[event_index] == current_event_number):  # Loop over other event hits
                if np.isfinite(position_array_x[index]) and np.isfinite(position_array_x[event_index]):
                    # check if distance is smaller than limit
                    if use_ellipse and max_distance_x > 0 and max_distance_y > 0:  # use ellipse
                        if ((np.square(position_array_x[index] - position_array_x[event_index]) / max_distance_x**2) + (np.square(position_array_y[index] - position_array_y[event_index]) / max_distance_y**2)) <= 1:
                            small_distance_flag_array[index] = 1
                            small_distance_flag_array[event_index] = 1
                    else:  # use square
                        if (abs(position_array_x[index] - position_array_x[event_index]) <= max_distance_x) and (abs(position_array_y[index] - position_array_y[event_index]) <= max_distance_y):
                            small_distance_flag_array[index] = 1
                            small_distance_flag_array[event_index] = 1
                event_index += 1
            index += 1


def create_results_array(n_duts, dut_offsets, dut_slopes, track_chi2s, quality_flag, isolated_track_flag, isolated_hit_flag, select_fit_tracks, select_tracks_for_storage, track_candidates_chunk, keep_data, track_estimates_chunk_full):
    select_fitted_tracks_for_storage = select_tracks_for_storage[select_fit_tracks]
    # Tracks description, additional columns
    tracks_descr = []
    for dimension in ['x', 'y', 'z']:
        tracks_descr.append(('offset_%s' % dimension, track_candidates_chunk["x_dut_0"].dtype))
    for dimension in ['x', 'y', 'z']:
        tracks_descr.append(('slope_%s' % dimension, track_candidates_chunk["x_dut_0"].dtype))
    if track_estimates_chunk_full is not None:
        for index_dut in range(n_duts):
            for index in ['offset', 'slope']:
                for dimension in ['x', 'y', 'z']:
                    tracks_descr.append(('%s_%s_dut_%d' % (index, dimension, index_dut), track_candidates_chunk["x_dut_0"].dtype))
    tracks_descr.extend([('track_chi2', np.float64),
                         ('quality_flag', track_candidates_chunk["hit_flag"].dtype),
                         ('isolated_track_flag', track_candidates_chunk["hit_flag"].dtype),
                         ('isolated_hit_flag', track_candidates_chunk["hit_flag"].dtype)])

    # Select only fitted tracks (keep_data is False) or keep all track candidates (keep_data is True)
    if not keep_data:
        track_candidates_chunk = track_candidates_chunk[select_tracks_for_storage]

    tracks_array = np.empty((track_candidates_chunk.shape[0],), dtype=track_candidates_chunk.dtype.descr + tracks_descr)

    tracks_array['hit_flag'] = track_candidates_chunk['hit_flag']
    tracks_array['event_number'] = track_candidates_chunk['event_number']
    for index_dut in range(n_duts):
        tracks_array['x_dut_%d' % index_dut] = track_candidates_chunk['x_dut_%d' % index_dut]
        tracks_array['y_dut_%d' % index_dut] = track_candidates_chunk['y_dut_%d' % index_dut]
        tracks_array['z_dut_%d' % index_dut] = track_candidates_chunk['z_dut_%d' % index_dut]
        tracks_array['x_err_dut_%d' % index_dut] = track_candidates_chunk['x_err_dut_%d' % index_dut]
        tracks_array['y_err_dut_%d' % index_dut] = track_candidates_chunk['y_err_dut_%d' % index_dut]
        tracks_array['z_err_dut_%d' % index_dut] = track_candidates_chunk['z_err_dut_%d' % index_dut]
        tracks_array['charge_dut_%d' % index_dut] = track_candidates_chunk['charge_dut_%d' % index_dut]
        tracks_array['frame_dut_%d' % index_dut] = track_candidates_chunk['frame_dut_%d' % index_dut]
        tracks_array['n_hits_dut_%d' % index_dut] = track_candidates_chunk['n_hits_dut_%d' % index_dut]
        tracks_array['cluster_ID_dut_%d' % index_dut] = track_candidates_chunk['cluster_ID_dut_%d' % index_dut]
        tracks_array['cluster_shape_dut_%d' % index_dut] = track_candidates_chunk['cluster_shape_dut_%d' % index_dut]
        tracks_array['n_cluster_dut_%d' % index_dut] = track_candidates_chunk['n_cluster_dut_%d' % index_dut]
        tracks_array['tdc_value_dut_%d' % index_dut] = track_candidates_chunk['tdc_value_dut_%d' % index_dut]
        tracks_array['tdc_timestamp_dut_%d' % index_dut] = track_candidates_chunk['tdc_timestamp_dut_%d' % index_dut]
        tracks_array['tdc_status_dut_%d' % index_dut] = track_candidates_chunk['tdc_status_dut_%d' % index_dut]

    if keep_data:
        for index, dimension in enumerate(['x', 'y', 'z']):
            tracks_array['offset_%s' % dimension][select_fit_tracks] = dut_offsets[:, index]
            tracks_array['slope_%s' % dimension][select_fit_tracks] = dut_slopes[:, index]
            tracks_array['offset_%s' % dimension][~select_fit_tracks] = np.nan
            tracks_array['slope_%s' % dimension][~select_fit_tracks] = np.nan
        if track_estimates_chunk_full is not None:
            for index_dut in range(n_duts):
                tracks_array['offset_x_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 0]
                tracks_array['offset_y_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 1]
                tracks_array['offset_z_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 2]
                tracks_array['slope_x_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 3]
                tracks_array['slope_y_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 4]
                tracks_array['slope_z_dut_%d' % index_dut][select_fit_tracks] = track_estimates_chunk_full[:, index_dut, 5]
                tracks_array['offset_x_dut_%d' % index_dut][~select_fit_tracks] = np.nan
                tracks_array['offset_y_dut_%d' % index_dut][~select_fit_tracks] = np.nan
                tracks_array['offset_z_dut_%d' % index_dut][~select_fit_tracks] = np.nan
                tracks_array['slope_x_dut_%d' % index_dut][~select_fit_tracks] = np.nan
                tracks_array['slope_y_dut_%d' % index_dut][~select_fit_tracks] = np.nan
                tracks_array['slope_z_dut_%d' % index_dut][~select_fit_tracks] = np.nan
        tracks_array['track_chi2'][select_fit_tracks] = track_chi2s
        tracks_array['track_chi2'][~select_fit_tracks] = np.nan
        tracks_array['quality_flag'][select_fit_tracks] = quality_flag
        tracks_array['quality_flag'][~select_fit_tracks] = 0
        tracks_array['isolated_track_flag'][select_fit_tracks] = isolated_track_flag
        tracks_array['isolated_track_flag'][~select_fit_tracks] = 0
        tracks_array['isolated_hit_flag'][select_fit_tracks] = isolated_hit_flag
        tracks_array['isolated_hit_flag'][~select_fit_tracks] = 0
    else:
        for index, dimension in enumerate(['x', 'y', 'z']):
            tracks_array['offset_%s' % dimension] = dut_offsets[select_fitted_tracks_for_storage, index]
            tracks_array['slope_%s' % dimension] = dut_slopes[select_fitted_tracks_for_storage, index]
        if track_estimates_chunk_full is not None:
            for index_dut in range(n_duts):
                tracks_array['offset_x_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 0]
                tracks_array['offset_y_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 1]
                tracks_array['offset_z_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 2]
                tracks_array['slope_x_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 3]
                tracks_array['slope_y_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 4]
                tracks_array['slope_z_dut_%d' % index_dut] = track_estimates_chunk_full[select_fitted_tracks_for_storage, index_dut, 5]
        tracks_array['track_chi2'] = track_chi2s[select_fitted_tracks_for_storage]
        tracks_array['quality_flag'] = quality_flag[select_fitted_tracks_for_storage]
        tracks_array['isolated_track_flag'] = isolated_track_flag[select_fitted_tracks_for_storage]
        tracks_array['isolated_hit_flag'] = isolated_hit_flag[select_fitted_tracks_for_storage]

    return tracks_array


def _fit_tracks_loop(track_hits):
    '''
    Loop over the selected tracks.

    Parameters
    ----------
    track_hits : array
        Array which contains the x, y and z hit position of each DUT for all tracks.

    Returns
    -------
    offset : array
        Array, which contains the track offsets.
    slope : array
        Array, which contains the track slopes.
    '''
    slope = np.full((track_hits.shape[0], 3), dtype=np.float64, fill_value=np.nan)
    offset = np.full((track_hits.shape[0], 3), dtype=np.float64, fill_value=np.nan)

    # Loop over selected track candidate hits and fit
    for index, hits in enumerate(track_hits):
        try:
            offset[index], slope[index] = line_fit_3d(positions=hits)
        except np.linalg.linalg.LinAlgError:
            pass

    return offset, slope


def line_fit_3d(positions):
    '''
    3D line fit (SVD).
    '''
    # remove NaNs from data
    positions = positions[~np.isnan(positions).any(axis=1)]
    # calculating offset and slope for given set of 3D points
    # calculate mean to subtract mean for each component (x,y,z) for SVD calculation
    offset = positions.mean(axis=0)
    # TODO: mean calculation and substraction can be raplced with svd(cov(points))
    slope = np.linalg.svd(positions - offset, full_matrices=False)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    # normalize to 1
    slope_mag = np.sqrt(slope.dot(slope))
    slope /= slope_mag
    # force the 3rd component (z) to be positive
    if slope[2] < 0:
        slope = -slope
    # intersections = offset + slope / slope[2] * (positions.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
    # # calculate the sum of the squared x/y residuals
    # chi2 = np.sum(np.square(positions - intersections))
    return offset, slope  # , chi2


def _fit_tracks_kalman_loop(track_hits, telescope, select_fit_duts, beam_energy, particle_mass, scattering_planes):
    '''
    Loop over the selected tracks. In this function all matrices for the Kalman Filter are calculated track by track
    and the Kalman Filter is started.

    Parameters
    ----------
    track_hits : array
        Array which contains the x/y/z hit position and error for all DUTs and all tracks.
    telsescope : object
        Telescope object.
    select_fit_duts : list
        The select_fit_duts is a subset of all DUT indices. A DUT that is not included, will be omitted during the filtering step.
    beam_energy : float
        Energy of the beam in MeV, e.g., 2500.0 MeV for ELSA beam.
    particle_mass : float
        Mass of the particle in MeV, e.g., 0.511 MeV for electrons.
    scattering_planes : list or dict
        Specifies additional scattering planes in case of DUTs which are not used or additional material in the way of the tracks.
        The list must contain dictionaries containing the following keys:
            material_budget: material budget of the scattering plane
            translation_x/translation_y/translation_z: x/y/z position of the plane (in um)
            rotation_alpha/rotation_beta/rotation_gamma: alpha/beta/gamma angle of scattering plane (in radians)
        If scattering_planes is None, no scattering plane will be added.

    Returns
    -------
    smoothed_state_estimates : array_like
        Smoothed state vectors, which contains (smoothed x position, smoothed y position, slope_x, slope_y).
    chi2 : uint
        Chi2 of track.
    x_err : array_like
        Error of smoothed hit position in x direction. Calculated from smoothed
        state covariance matrix. Only approximation, since only diagonal element is taken.
    y_err : array_like
        Error of smoothed hit position in y direction. Calculated from smoothed
        state covariance matrix. Only approximation, since only diagonal element is taken.
    '''
    if scattering_planes is None:
        scattering_planes = []
    elif isinstance(scattering_planes, dict):
        scattering_planes = [scattering_planes]
    alignment = []
    material_budget = []
    all_dut_planes = [dut for dut in telescope]
    all_dut_planes.extend(scattering_planes)
    for dut in all_dut_planes:
        alignment.append([dut.translation_x, dut.translation_y, dut.translation_z, dut.rotation_alpha, dut.rotation_beta, dut.rotation_gamma])
        # TODO: take rotation into account for material budget
        material_budget.append(dut.material_budget)
    alignment = np.array(alignment)
    material_budget = np.array(material_budget)

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in all_dut_planes:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)
    # z_sorted_fit_dut_indices = []
    # for dut_index in z_sorted_dut_indices:
    #     if dut_index in select_fit_duts:
    #         z_sorted_fit_dut_indices.append(dut_index)

    # TODO: check if calculation of pixel size is necessary
    # try:
    #     first_dut = telescope[z_sorted_dut_indices[0]]
    #     first_dut_x_pixel_size, first_dut_y_pixel_size, _ = np.abs(first_dut.local_to_global_position(
    #         x=[first_dut.column_size],
    #         y=[first_dut.row_size],
    #         z=[0.0],
    #         translation_x=0.0,
    #         translation_y=0.0,
    #         translation_z=0.0))
    #     first_dut_pixel_size = [first_dut_x_pixel_size[0], first_dut_y_pixel_size[0]]
    # except AttributeError:  # First plane is scattering plane
    #     first_dut_pixel_size = [0.0, 0.0]

    if scattering_planes:
        track_hits = np.append(arr=track_hits, values=np.full((track_hits.shape[0], track_hits.shape[2] * len(scattering_planes)), fill_value=np.nan, dtype=np.float64), axis=1)

    chunk_size = track_hits.shape[0]
    n_duts = len(all_dut_planes)

    # Calculate multiple scattering
    momentum = np.sqrt(beam_energy**2 - particle_mass**2)
    beta = momentum / beam_energy  # almost 1

    if np.any(np.isclose(material_budget[z_sorted_dut_indices[:-1]], 0.0)):
        raise ValueError("Material budget is zero.")

    # rms angle of multiple scattering
    thetas = np.array(((13.6 / momentum / beta) * np.sqrt(material_budget) * (1. + 0.038 * np.log(material_budget))))
    # error on z-position
    z_error = 1e3  # Assume 1 mm error on z-position (alignment, precision of setup,...)

    # express transition and observation offset matrices
    # these are additional offsets, which are not used at the moment
    transition_offsets = np.full((chunk_size, n_duts, 6), fill_value=np.nan, dtype=np.float64)
    transition_offsets[:, z_sorted_dut_indices[:-1], :] = 0.0
    observation_offsets = np.zeros((chunk_size, n_duts, 3), dtype=np.float64)

    # express initial state. Contains (x_pos, y_pos, z_pos, slope_x, slope_y, slope_z).
    initial_state_mean = np.zeros((chunk_size, 6), dtype=np.float64)

    # express observation matrix, only observe (x,y,z)
    observation_matrices = np.zeros((chunk_size, n_duts, 3, 6), dtype=np.float64)
    observation_matrices[:, :, 0, 0] = 1.0
    observation_matrices[:, :, 1, 1] = 1.0
    observation_matrices[:, :, 2, 2] = 1.0
    # express observation covariance matrices
    observation_covariances = np.zeros((chunk_size, n_duts, 3, 3), dtype=np.float64)

    # express initial state covariance matrices
    initial_state_covariance = np.zeros((chunk_size, 6, 6), dtype=np.float64)
    # error on initial slope is roughly divergence of beam (5 mrad).
    initial_state_covariance[:, 3, 3] = np.square(5e-3)
    initial_state_covariance[:, 4, 4] = np.square(5e-3)
    initial_state_covariance[:, 5, 5] = np.square(5e-3)

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        # Take cluster hit position error as measurement error for duts which have a hit.
        # For those who have no hit, need no error, since the should not be included in fit via fit selection
        duts_with_hits = np.array(range(n_duts), dtype=np.int)[~np.isnan(actual_hits[:, 0])]
        observation_covariances[index, duts_with_hits, 0, 0] = np.square(actual_hits[duts_with_hits, 3])
        observation_covariances[index, duts_with_hits, 1, 1] = np.square(actual_hits[duts_with_hits, 4])
        observation_covariances[index, duts_with_hits, 2, 2] = np.square(z_error)  # Assume 1 mm error on z-position measurement

        if np.isnan(actual_hits[z_sorted_dut_indices[0], 0]):  # The first plane has no hit
            # Take planes from fit selction and fit a line to the hits,
            # then extrapolate the line to first plane in order to find initial state.
            # The position error is estimated with the pixel size.
            # TODO: Can't we handle this as any other scattering plane with error=0?
            # Edit: Any plane without hit is treated as scatter plane.
            try:
                # Fit all DUTs with hits
                offset, slope = line_fit_3d(positions=actual_hits[select_fit_duts, :3])
                # TODO: For lower energies and lighter particles use the first hit DUT as position for the first scatter plane
                # Fit the first 2 DUTs with hits
                # offset, slope = line_fit_3d(positions=actual_hits[z_sorted_fit_dut_indices, :3], n=2)
            except np.linalg.linalg.LinAlgError:
                offset, slope = np.nan, np.nan

            intersections = geometry_utils.get_line_intersections_with_dut(
                line_origins=np.array([offset]),
                line_directions=np.array([slope]),
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [intersections[0, 0], intersections[0, 1], intersections[0, 2], 0.0, 0.0, 1.0]
            # initial_state_covariance[index, 0, 0] = np.square(x_error_init_no_hit)
            # initial_state_covariance[index, 1, 1] = np.square(y_error_init_no_hit)
            initial_state_covariance[index, 0, 0] = np.square(dut.pixel_size[0])
            initial_state_covariance[index, 1, 1] = np.square(dut.pixel_size[1])
            initial_state_covariance[index, 2, 2] = np.square(z_error)
        else:  # The first plane has a hit
            # If first plane should be included in track building, take first dut hit as initial value and
            # its corresponding cluster position error as the error on the measurement.
            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [actual_hits[z_sorted_dut_indices[0], 0], actual_hits[z_sorted_dut_indices[0], 1], actual_hits[z_sorted_dut_indices[0], 2], 0.0, 0.0, 1.0]
            initial_state_covariance[index, 0, 0] = np.square(actual_hits[z_sorted_dut_indices[0], 3])  # x_err
            initial_state_covariance[index, 1, 1] = np.square(actual_hits[z_sorted_dut_indices[0], 4])  # y_err
            initial_state_covariance[index, 2, 2] = np.square(z_error)

    # Do some sanity check: Covariance matrices should be positive semi-definite (all eigenvalues have to be non-negative)
    for cov in [observation_covariances, initial_state_covariance]:
        if not np.all(np.linalg.eigvalsh(cov) >= 0.0):
            raise RuntimeError('Covariance matrices are not positive semi-definite!')

    # run kalman filter
    track_estimates_chunk, x_err, y_err, chi2 = _kalman_fit_3d(
        dut_planes=all_dut_planes,
        z_sorted_dut_indices=z_sorted_dut_indices,
        hits=track_hits[:, :, 0:3],
        thetas=thetas,
        select_fit_duts=select_fit_duts,
        transition_offsets=transition_offsets,
        observation_matrices=observation_matrices,
        observation_covariances=observation_covariances,
        observation_offsets=observation_offsets,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance)

    # remove scatter planes from data
    x_err = x_err[:, :len(telescope)]
    y_err = y_err[:, :len(telescope)]

    offsets = track_estimates_chunk[:, :len(telescope), :3]

    slopes = track_estimates_chunk[:, :len(telescope), 3:]
    # force the 3rd component (z) to be positive
    # and normalize to 1
    slopes[slopes[:, :, 2] < 0.0] = -slopes[slopes[:, :, 2] < 0.0]
    slopes_mag = np.sqrt(np.einsum('ijk,ijk->ij', slopes, slopes))
    slopes /= slopes_mag[:, :, np.newaxis]

    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError("Not all chi-square values are positive!")

    # Sum up all chi2 and divide by number of degrees of freedom.
    # chi2 = np.nansum(chi2, axis=1) / np.count_nonzero(~np.isnan(chi2), axis=1)
    chi2 = np.nansum(chi2[:, select_fit_duts], axis=1) / (3 * (np.count_nonzero(~np.isnan(chi2[:, select_fit_duts]), axis=1) - 3))

    return offsets, slopes, chi2, x_err, y_err


def _kalman_fit_3d(dut_planes, z_sorted_dut_indices, hits, thetas, select_fit_duts, transition_offsets, observation_matrices, observation_covariances, observation_offsets, initial_state_mean, initial_state_covariance):
    '''
    This function calls the Kalman Filter. It returns track by track the smoothed state vector which contains in the first two components
    the smoothed hit positions and in the last two components the respective slopes. Additionally the chi square of the track is calculated
    and returned.

    Parameters
    ----------
    dut_planes : list
        List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
    z_sorted_dut_indices : list
        List of DUT indices in the order reflecting their z position.
    hits : array_like
        Array which contains the x, y and z hit position of each DUT for one track.
    thetas : list
        List of scattering angle root mean squares (RMS).
    select_fit_duts : list
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    transition_offset : array_like
        Vector which array_like the offset of each transition.
    observation_matrix : array_like
        Matrix which converts the state vector to the actual measurement vector.
    observation_covariances : array_like
        Matrix which describes the covariance of the measurement.
    observation_offset : array_like
        Vector which describes the offset of each measurement.
    initial_state_mean : array_like
        Vector which describes the starting point of the state vector.
    initial_state_covariance : array_like
        Error on the starting pointin of the state vector.

    Returns
    -------
    smoothed_state_estimates : array_like
        Smoothed state vectors.
    chi2 : uint
        Chi2 of track.
    x_err : array_like
        Error of smoothed hit position in x direction. Calculated from smoothed
        state covariance matrix.
    y_err : array_like
        Error of smoothed hit position in y direction. Calculated from smoothed
        state covariance matrix.
    '''
    kf = kalman.KalmanFilter()
    smoothed_state_estimates, cov, chi2 = kf.smooth(
        dut_planes=dut_planes,
        z_sorted_dut_indices=z_sorted_dut_indices,
        observations=hits[:, :, 0:3],
        thetas=thetas,
        select_fit_duts=select_fit_duts,
        transition_offsets=transition_offsets,
        observation_matrices=observation_matrices,
        observation_offsets=observation_offsets,
        observation_covariances=observation_covariances,
        initial_state=initial_state_mean,
        initial_state_covariance=initial_state_covariance)

    # calculate the sum of the squared x/y residuals, divided by n hits per track
    # chi2 = np.sum(np.square(np.ma.masked_invalid(hits[:, :, 0:2]) - smoothed_state_estimates[:, :, 0:2]), dtype=np.float64, axis=(1, 2)) / np.count_nonzero(~np.isnan(hits[:, :, 0]), axis=1)

    # rough estimate for error on x and y of smoothed estimate
    x_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2)[:, :, 0])
    y_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2)[:, :, 1])

    # Check for invalid values (NaN)
    if np.any(np.isnan(smoothed_state_estimates)):
        logging.warning('Smoothed state estimates contain invalid values (NaNs). Check input of Kalman Filter.')

    return smoothed_state_estimates, x_err, y_err, chi2

def combinatorial_kalman_filter(telescope_configuration, input_merged_file, output_tracks_file, max_events=None, select_duts=None, select_hit_duts=None, select_fit_duts=None, exclude_dut_hit=True, min_track_hits=None, beam_energy=None, particle_mass=None, quality_distances=(250.0, 250.0), reject_quality_distances=(500.0, 500.0), association_distances=(500, 500), max_hit_multiplicities=[3], max_delta_chi=15, align_to_beam=True, use_limits=True, full_track_info=False, plot=True, chunk_size=50000):
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Running Combinatorial Kalman Filter of %d DUTs ===' % n_duts)

    if output_tracks_file is None:
        output_tracks_file = os.path.join(os.path.dirname(input_merged_file), 'Tracks_CKF.h5')
    if plot:
        output_pdf_file = os.path.splitext(output_tracks_file)[0] + '.pdf'
        output_pdf = PdfPages(output_pdf_file, keep_empty=False)
    else:
        output_pdf = None

    if select_duts is None:
        select_duts = range(n_duts)  # standard setting: fit tracks for all DUTs
    elif not isinstance(select_duts, Iterable):
        select_duts = [select_duts]
    # Check for duplicates
    if len(select_duts) != len(set(select_duts)):
        raise ValueError("found douplicate in select_duts")
    # Check if any iterable in iterable
    if any(map(lambda val: isinstance(val, Iterable), select_duts)):
        raise ValueError("item in select_duts is iterable")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = range(n_duts)
#         # copy each item
#         for hit_duts in select_hit_duts:
#             select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("select_fit_duts is no iterable")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("select_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("not all items in select_fit_duts are iterable")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_fit_duts has the wrong length")
    for index, fit_dut in enumerate(select_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("item in select_fit_duts has length < 2")
#         if set(fit_dut) - set(select_hit_duts[index]):  # fit DUTs are required to have a hit
#             raise ValueError("DUT in select_fit_duts is not in select_hit_duts")

    # Create track, hit selection
    if select_hit_duts is None:  # If None, require no hit
        #         select_hit_duts = range(n_duts)
        select_hit_duts = []
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("select_hit_duts is no iterable")
#     elif not select_hit_duts:  # empty iterable
#         raise ValueError("select_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("not all items in select_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_hit_duts has the wrong length")
#     for hit_dut in select_hit_duts:
#         if len(hit_dut) < 2:  # check the length of the items
#             raise ValueError("item in select_hit_duts has length < 2")

    # Create quality distance
    if isinstance(quality_distances, tuple) or quality_distances is None:
        quality_distances = [quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(quality_distances, Iterable):
        raise ValueError("quality_distances is no iterable")
    elif not quality_distances:  # empty iterable
        raise ValueError("quality_distances has no items")
    # Finally check length of all arrays
    if len(quality_distances) != n_duts:  # empty iterable
        raise ValueError("quality_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, quality_distances)):
        raise ValueError("not all items in quality_distances are iterable")
    # Finally check length of all arrays
    for distance in quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in quality_distances has length != 2")

    # Create reject quality distance
    if isinstance(reject_quality_distances, tuple) or reject_quality_distances is None:
        reject_quality_distances = [reject_quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(reject_quality_distances, Iterable):
        raise ValueError("reject_quality_distances is no iterable")
    elif not reject_quality_distances:  # empty iterable
        raise ValueError("reject_quality_distances has no items")
    # Finally check length of all arrays
    if len(reject_quality_distances) != n_duts:  # empty iterable
        raise ValueError("reject_quality_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, reject_quality_distances)):
        raise ValueError("not all items in reject_quality_distances are iterable")
    # Finally check length of all arrays
    for distance in reject_quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in reject_quality_distances has length != 2")

    # Create association distances
    if isinstance(association_distances, tuple) or association_distances is None:
        association_distances = [association_distances] * n_duts
    # Check iterable and length
    if not isinstance(association_distances, Iterable):
        raise ValueError("association_distances is no iterable")
    elif not association_distances:  # empty iterable
        raise ValueError("association_distances has no items")
    # Finally check length of all arrays
    if len(association_distances) != n_duts:  # empty iterable
        raise ValueError("association_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, association_distances)):
        raise ValueError("not all items in association_distances are iterable")
    # Finally check length of all arrays
    for distance in association_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in association_distances has length != 2")

    # Create max_hit_multiplicities
    if not isinstance(max_hit_multiplicities, Iterable):
        max_hit_multiplicities = [max_hit_multiplicities] * n_duts
    # Check iterable and length
    if not isinstance(max_hit_multiplicities, Iterable):
        raise ValueError("max_hit_multiplicities is no iterable")
    elif not max_hit_multiplicities:  # empty iterable
        raise ValueError("max_hit_multiplicities has no items")
    # Finally check length of all arrays
    if len(max_hit_multiplicities) != n_duts:  # empty iterable
        raise ValueError("max_hit_multiplicities has the wrong length")

    # Check iterable and length
    if not isinstance(exclude_dut_hit, Iterable):
        exclude_dut_hit = [exclude_dut_hit] * len(select_duts)
    elif not exclude_dut_hit:  # empty iterable
        raise ValueError("exclude_dut_hit has no items")
    # Finally check length of all array
    if len(exclude_dut_hit) != len(select_duts):  # empty iterable
        raise ValueError("exclude_dut_hit has the wrong length")
    # Check if only bools in iterable
    if not all(map(lambda val: isinstance(val, (bool,)), exclude_dut_hit)):
        raise ValueError("not all items in exclude_dut_hit are boolean")

    pool = Pool()  # For multiprocessing
    fitted_duts = []
    # Load hits from merged file
    with tb.open_file(input_merged_file, mode='r') as in_file_h5:
        tracklets_node = in_file_h5.root.MergedClusters

        with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
            for fit_dut_index, actual_fit_dut in enumerate(select_duts):  # Loop over the DUTs where tracks shall be fitted for
                if actual_fit_dut in fitted_duts:
                    continue
                # Test whether other DUTs have identical tracks
                # if yes, save some CPU time and fit only once.
                # This following list contains all DUT indices that will be fitted
                # during this step of the loop.
                actual_fit_duts = []
                for curr_fit_dut_index, curr_fit_dut in enumerate(select_duts):
                    if ((curr_fit_dut == actual_fit_dut) or (exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is False) and set(select_hit_duts[curr_fit_dut_index]) == set(select_hit_duts[fit_dut_index]) and set(select_fit_duts[curr_fit_dut_index]) == set(select_fit_duts[fit_dut_index])):
                        actual_fit_duts.append(curr_fit_dut)
                logging.info('= Running Combinatorial Kalman Filter for %s =', ', '.join([telescope[curr_dut].name for curr_dut in actual_fit_duts]))
                # remove existing nodes
                for dut_index in actual_fit_duts:
                    try:  # Check if table already exists, then append data
                        out_file_h5.remove_node(out_file_h5.root, name='Tracks_DUT%d' % dut_index)
                        logging.info('Overwriting existing tracks for DUT%d', dut_index)
                    except tb.NodeError:  # Table does not exist, thus create new
                        pass
                total_n_tracks = tracklets_node.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0

                # select hit DUTs based on input parameters
                hit_duts = list(set(select_hit_duts[fit_dut_index]) - set(actual_fit_duts)) if exclude_dut_hit[fit_dut_index] else select_hit_duts[fit_dut_index]
                actual_min_track_hits = min_track_hits[fit_dut_index]
                can_have_hits = list(set(range(len(telescope))) - set(select_hit_duts[fit_dut_index]))  # DUTs which can have a hit (exclude DUTs which require a hit)

#                 if actual_min_track_hits is None:
#                     dut_hit_selection = 0  # DUTs required to have hits
#                     for dut_index in hit_duts:
#                         dut_hit_selection |= ((1 << dut_index))
#                 dut_hit_selection = [dut_hit_selection]
#                 else:
#                     dut_hit_selection = []
#                     all_dut_hit_selection = 0
#                     can_have_hits = list(set(range(len(telescope))) - set(select_hit_duts[fit_dut_index]))  # DUTs which can have a hit (exclude DUTs which require a hit)
#                     dut_hit_combinations = combinations(can_have_hits, actual_min_track_hits)  # Get all possible combinations of DUTs which can have a hit
# 
#                     all_hit_duts = can_have_hits + hit_duts  # Duts which are allowed to have a hit
#                     for index in all_hit_duts:
#                         all_dut_hit_selection |= (1 << index)
# 
#                     # Create dut hit selection
#                     for dut_hit_combination in list(dut_hit_combinations):  # Loop over all combinations
#                         hit_selection = all_dut_hit_selection  # Reset hit selection for every combination
#                         no_hit_duts = set(can_have_hits) - set(dut_hit_combination)
#                         for index in no_hit_duts:
#                             hit_selection -= (1 << index)  # Substract each DUT
#                         dut_hit_selection.append(hit_selection)
#                 if actual_min_track_hits is None:
                logging.info('Require hits in %d DUTs for track selection: %s', len(hit_duts), ', '.join([telescope[curr_dut].name for curr_dut in hit_duts]))
#                 else:
#                     logging.info('Require at least %d hits in the following DUTs for track selection: %s', actual_min_track_hits, ', '.join([telescope[curr_dut].name for curr_dut in can_have_hits]))
#                     logging.info('Require hits in %d DUTs for track selection: %s', len(hit_duts), ', '.join([telescope[curr_dut].name for curr_dut in hit_duts]))
                # select fit DUTs based on input parameters
                dut_fit_selection = 0  # DUTs to be used for the fit
                fit_duts = list(set(select_fit_duts[fit_dut_index]) - set(actual_fit_duts)) if exclude_dut_hit[fit_dut_index] else select_fit_duts[fit_dut_index]
                for dut_index in fit_duts:
                    dut_fit_selection |= ((1 << dut_index))
                logging.info("Use %d DUTs for track fit: %s", len(fit_duts), ', '.join([telescope[curr_dut].name for curr_dut in fit_duts]))
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=total_n_tracks,
                                                       term_width=80)
                # progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=max_tracks if max_tracks is not None else in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                progress_bar.start()

                chunk_indices = []
                chunk_stats = []
                dut_stats = []
                # Loop over tracklets
                for tracklets_data_chunk, index_chunk in analysis_utils.data_aligned_at_events(tracklets_node, start_event_number=0, stop_event_number=2500000, chunk_size=chunk_size):
                    chunk_indices.append(index_chunk)
                    chunk_stats.append(1.0)  # TODO: fix this
                    n_tracks_chunk = tracklets_data_chunk.shape[0]

                    unique_events = np.unique(tracklets_data_chunk["event_number"])
                    n_events_chunk = unique_events.shape[0]

                    if max_events:
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                        else:
                            # calculate correction of number of selected events
                            correction = (total_n_tracks - index_chunk)/total_n_tracks * 1 / (((total_n_tracks-last_index_chunk)/total_n_tracks)/((max_events-total_n_events_stored_last)/max_events)) \
                                         + (index_chunk)/total_n_tracks * 1 / (((last_index_chunk)/total_n_tracks)/((total_n_events_stored_last)/max_events))
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(select_n_events, max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        selected_tracks = np.in1d(tracklets_data_chunk["event_number"], selected_events)
                        store_n_tracks = np.count_nonzero(selected_tracks)
                        # TODO: total_n_tracks_stored not used...
                        total_n_tracks_stored += store_n_tracks
                        tracklets_data_chunk = tracklets_data_chunk[selected_tracks]

                    # Prepare hit data for track finding, create temporary arrays for x, y, z position and charge data
                    # This is needed to call a numba jitted function, since the number of DUTs is not fixed and thus the data format
                    track_hits = np.full((tracklets_data_chunk.shape[0], n_duts, 6), fill_value=np.nan, dtype=np.float)
                    for dut_index, dut in enumerate(telescope):  # Fill index loop of new array
                        # TODO: taking telescope alignment into account for initial state
                        # apply alignment for fitting the tracks
                        track_hits[:, dut_index, 0], track_hits[:, dut_index, 1], track_hits[:, dut_index, 2] = dut.local_to_global_position(
                            x=tracklets_data_chunk['x_dut_%s' % dut_index],
                            y=tracklets_data_chunk['y_dut_%s' % dut_index],
                            z=tracklets_data_chunk['z_dut_%s' % dut_index])
                        track_hits[:, dut_index, 3], track_hits[:, dut_index, 4], track_hits[:, dut_index, 5] = np.abs(dut.local_to_global_position(
                            x=tracklets_data_chunk['x_err_dut_%s' % dut_index],
                            y=tracklets_data_chunk['y_err_dut_%s' % dut_index],
                            z=tracklets_data_chunk['z_err_dut_%s' % dut_index],
                            # no translation for the errors
                            translation_x=0.0,
                            translation_y=0.0,
                            translation_z=0.0))
#                         if align_to_beam:
#                             track_hits[:, dut_index, 0], track_hits[:, dut_index, 1], track_hits[:, dut_index, 2] = dut.local_to_global_position(
#                                 x=track_hits[:, dut_index, 0],
#                                 y=track_hits[:, dut_index, 1],
#                                 z=track_hits[:, dut_index, 2],
#                                 translation_x=telescope.translation_x,
#                                 translation_y=telescope.translation_y,
#                                 translation_z=telescope.translation_z,
#                                 rotation_alpha=telescope.rotation_alpha,
#                                 rotation_beta=telescope.rotation_beta,
#                                 rotation_gamma=telescope.rotation_gamma)
# 
#                             track_hits[:, dut_index, 3], track_hits[:, dut_index, 4], track_hits[:, dut_index, 5] = np.abs(dut.local_to_global_position(
#                                 x=track_hits[:, dut_index, 3],
#                                 y=track_hits[:, dut_index, 4],
#                                 z=track_hits[:, dut_index, 5],
#                                 translation_x=0.0,
#                                 translation_y=0.0,
#                                 translation_z=0.0,
#                                 rotation_alpha=telescope.rotation_alpha,
#                                 rotation_beta=telescope.rotation_beta,
#                                 rotation_gamma=telescope.rotation_gamma))

                    per_event_track_variables = np.zeros(shape=(4, tracklets_data_chunk.shape[0]))
                    per_dut_track_variables = np.zeros(shape=(6, tracklets_data_chunk.shape[0], n_duts))
                    # Fill track variables (all others need to be still calculated)
                    per_event_track_variables[0] = tracklets_data_chunk['event_number']
                    for dut_index in range(n_duts):
                        per_dut_track_variables[0, :, dut_index] = tracklets_data_chunk['charge_dut_%d' % dut_index]
                        per_dut_track_variables[1, :, dut_index] = tracklets_data_chunk['n_hits_dut_%d' % dut_index]
                        per_dut_track_variables[2, :, dut_index] = tracklets_data_chunk['cluster_shape_dut_%d' % dut_index]
                        per_dut_track_variables[3, :, dut_index] = tracklets_data_chunk['n_cluster_dut_%d' % dut_index]
                        per_dut_track_variables[4, :, dut_index] = tracklets_data_chunk['frame_dut_%d' % dut_index]
                        per_dut_track_variables[5, :, dut_index] = tracklets_data_chunk['cluster_ID_dut_%d' % dut_index]

                    # Split data and fit on all available cores.
                    n_slices = cpu_count()
                    logging.info('Doing combinatorial kalman filter on %i cores', n_slices)
                    # Take care that no event is splitted
                    _, unique_indices = np.unique(tracklets_data_chunk['event_number'], return_index=True)
                    split_indices = np.array_split(unique_indices, n_slices)
                    # track_hits_slices = np.array_split(track_hits, n_slices)
                    track_hits_slices = np.array_split(track_hits, [split_index[0] for split_index in split_indices[1:]])
                    per_event_track_variables_slices = np.array_split(per_event_track_variables, [split_index[0] for split_index in split_indices[1:]], axis=1)
                    per_dut_track_variables_slices = np.array_split(per_dut_track_variables, [split_index[0] for split_index in split_indices[1:]], axis=1)
   
                    results = [pool.apply_async(_combinatorial_kalman_filter, kwds={
                        'track_hits': track_hits_slice,
                        'per_event_track_variables': per_event_track_variables_slice,
                        'per_dut_track_variables': per_dut_track_variables_slice,
                        'telescope': telescope,
                        'select_fit_duts': fit_duts,
                        'select_hit_duts': hit_duts,
                        'can_have_hits': can_have_hits,
                        'max_missing_hits': len(can_have_hits) - actual_min_track_hits,
                        'max_hit_multiplicities': max_hit_multiplicities,
                        'association_distances': association_distances,
                        'quality_distances': quality_distances,
                        'beam_energy': beam_energy,
                        'particle_mass': particle_mass,
                        'max_delta_chi': max_delta_chi}) for (track_hits_slice, per_event_track_variables_slice, per_dut_track_variables_slice) in zip(track_hits_slices, per_event_track_variables_slices, per_dut_track_variables_slices)]

#                     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#                     fig = plt.figure()
#                     ax = fig.add_subplot(111, projection='3d')
#                     for i in range(track_hits.shape[0]):
#                         for dut_index in range(8):
#                             ax.scatter(track_hits[i, dut_index, 0], track_hits[i, dut_index, 1], track_hits[i, dut_index, 2], marker='<', label=('actual track hits %i' % dut_index) if i==0 else '')
#                     ax.set_xlabel('X Label')
#                     ax.set_ylabel('Y Label')
#                     ax.set_zlabel('Z Label')
#                     ax.set_title('CKF')
#                     plt.legend()
#                     plt.show()
#                     # Do combinatorial Kalman Filter. Returns final track hits and the smoothed states (Unique hit assignment ensured, tracks always selected based on lowest chi2, from smoothed result).
#                     (final_track_hits, track_estimates_chunk, final_per_event_track_variables, final_per_dut_track_variables) = _combinatorial_kalman_filter(
#                         # Possible track hits and corresponding event number
#                         track_hits=track_hits,
#                         per_event_track_variables=per_event_track_variables,
#                         per_dut_track_variables=per_dut_track_variables,
#                         telescope=telescope,
#                         # Settings
#                         select_fit_duts=fit_duts,
#                         select_hit_duts=hit_duts,
#                         can_have_hits=can_have_hits,
#                         max_missing_hits=len(can_have_hits) - actual_min_track_hits,
#                         max_hit_multiplicities=max_hit_multiplicities,
#                         association_distances=association_distances,
#                         quality_distances=quality_distances,
#                         beam_energy=beam_energy,
#                         particle_mass=particle_mass,
#                         max_delta_chi=max_delta_chi)

                    # Unpack result from multiprocessing
                    final_track_hits = np.concatenate([result.get()[0] for result in results])  # Merge offsets from all cores in results
                    track_estimates_chunk = np.concatenate([result.get()[1] for result in results])  # Merge slopes from all cores in results
                    final_per_event_track_variables = np.concatenate([result.get()[2] for result in results], axis=1)  # Merge event variables from all cores in results
                    final_per_dut_track_variables = np.concatenate([result.get()[3] for result in results], axis=1)  # Merge dut variables from all cores in results

                    # Put track candidates into same array structure as TrackCandidates. Therefore can use existing storing function.
                    track_candidates_array = np.empty((final_track_hits.shape[0],), dtype=tracklets_data_chunk.dtype.descr)
                    track_candidates_array['hit_flag'] = final_per_event_track_variables[2]
                    track_candidates_array['event_number'] = final_per_event_track_variables[0]
                    for dut_index, dut in enumerate(telescope):
                        # Transform hits into local coordinate system and write to result array
                        final_track_hits[:, dut_index, 0], final_track_hits[:, dut_index, 1], final_track_hits[:, dut_index, 2] = dut.global_to_local_position(
                            x=final_track_hits[:, dut_index, 0],
                            y=final_track_hits[:, dut_index, 1],
                            z=final_track_hits[:, dut_index, 2])
                        final_track_hits[:, dut_index, 3], final_track_hits[:, dut_index, 4], final_track_hits[:, dut_index, 5] = np.abs(dut.global_to_local_position(
                            x=final_track_hits[:, dut_index, 3],
                            y=final_track_hits[:, dut_index, 4],
                            z=final_track_hits[:, dut_index, 5],
                            # no translation for the errors
                            translation_x=0.0,
                            translation_y=0.0,
                            translation_z=0.0))
#                         if align_to_beam:
#                             final_track_hits[:, dut_index, 0], final_track_hits[:, dut_index, 1], final_track_hits[:, dut_index, 2] = dut.global_to_local_position(
#                                 x=final_track_hits[:, dut_index, 0],
#                                 y=final_track_hits[:, dut_index, 1],
#                                 z=final_track_hits[:, dut_index, 2],
#                                 translation_x=telescope.translation_x,
#                                 translation_y=telescope.translation_y,
#                                 translation_z=telescope.translation_z,
#                                 rotation_alpha=telescope.rotation_alpha,
#                                 rotation_beta=telescope.rotation_beta,
#                                 rotation_gamma=telescope.rotation_gamma)
#                             final_track_hits[:, dut_index, 3], final_track_hits[:, dut_index, 4], final_track_hits[:, dut_index, 5] = np.abs(dut.global_to_local_position(
#                                 x=final_track_hits[:, dut_index, 3],
#                                 y=final_track_hits[:, dut_index, 4],
#                                 z=final_track_hits[:, dut_index, 5],
#                                 translation_x=0.0,
#                                 translation_y=0.0,
#                                 translation_z=0.0,
#                                 rotation_alpha=telescope.rotation_alpha,
#                                 rotation_beta=telescope.rotation_beta,
#                                 rotation_gamma=telescope.rotation_gamma))

                        track_candidates_array['x_dut_%d' % dut_index] = final_track_hits[:, dut_index, 0]
                        track_candidates_array['y_dut_%d' % dut_index] = final_track_hits[:, dut_index, 1]
                        track_candidates_array['z_dut_%d' % dut_index] = final_track_hits[:, dut_index, 2]
                        track_candidates_array['x_err_dut_%d' % dut_index] = final_track_hits[:, dut_index, 3]
                        track_candidates_array['y_err_dut_%d' % dut_index] = final_track_hits[:, dut_index, 4]
                        track_candidates_array['z_err_dut_%d' % dut_index] = final_track_hits[:, dut_index, 5]
                        track_candidates_array['charge_dut_%d' % dut_index] = final_per_dut_track_variables[0, :, dut_index]
                        track_candidates_array['n_hits_dut_%d' % dut_index] = final_per_dut_track_variables[1, :, dut_index]
                        track_candidates_array['cluster_shape_dut_%d' % dut_index] = final_per_dut_track_variables[2, :, dut_index]
                        track_candidates_array['n_cluster_dut_%d' % dut_index] = final_per_dut_track_variables[3, :, dut_index]
                        track_candidates_array['frame_dut_%d' % dut_index] = final_per_dut_track_variables[4, :, dut_index]
                        track_candidates_array['cluster_ID_dut_%d' % dut_index] = final_per_dut_track_variables[5, :, dut_index]

                    dut_stats.append(store_track_data(
                        out_file_h5=out_file_h5,
                        track_candidates_chunk=track_candidates_array,
                        good_track_selection=np.full(shape=(track_candidates_array.shape[0]), fill_value=True, dtype=np.bool),
                        telescope=telescope,
                        offsets=track_estimates_chunk[:, :, :3],
                        slopes=track_estimates_chunk[:, :, 3:],
                        track_chi2s=final_per_event_track_variables[1],  # chi2
                        fit_duts=actual_fit_duts,  # storing tracks for these DUTs
                        select_fit_duts=fit_duts,  # DUTs used for fitting tracks
                        select_align_duts=None,  # Does not exist in CKF
                        quality_distances=quality_distances,
                        reject_quality_distances=reject_quality_distances,
                        use_limits=use_limits,
                        keep_data=False,  # Does not exist in CKF
                        method='ckf',
                        full_track_info=full_track_info))

                    # total_n_tracks += n_good_tracks
                    total_n_events_stored_last = total_n_events_stored
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
                    # progress_bar.update(min(total_n_tracks, max_tracks) if max_tracks is not None else index_chunk)
                progress_bar.finish()
                # print "***************"
                # print "total_n_tracks_stored", total_n_tracks_stored
                # print "total_n_events_stored", total_n_events_stored
                fitted_duts.extend(actual_fit_duts)

                plot_utils.plot_fit_tracks_statistics(
                    telescope=telescope,
                    fit_duts=actual_fit_duts,
                    chunk_indices=chunk_indices,
                    chunk_stats=chunk_stats,
                    dut_stats=dut_stats,
                    output_pdf=output_pdf)

    pool.close()
    pool.join()

    if output_pdf is not None:
        output_pdf.close()

    if plot:
        plot_utils.plot_track_chi2(input_tracks_file=output_tracks_file, output_pdf_file=None, dut_names=telescope.dut_names, chunk_size=chunk_size)

    return output_tracks_file


def _combinatorial_kalman_filter(track_hits, per_event_track_variables, per_dut_track_variables, telescope, select_fit_duts, select_hit_duts, can_have_hits, max_missing_hits, max_hit_multiplicities, association_distances, quality_distances, beam_energy, particle_mass, max_delta_chi):
    n_duts = track_hits.shape[1]
    n_dim_state = track_hits.shape[2]
    n_dim_obs = 3  # x,y,z as observations (as in KF)
#     # Needed arrays in order to store result of the Kalman Filter (after processing all planes) in order to use them for the Kalman Smoother
#     predicted_states_combined = np.empty((0, n_duts, n_dim_state))
#     predicted_state_covariances_combined = np.empty((0, n_duts, n_dim_state, n_dim_state))
#     filtered_states_combined = np.empty((0, n_duts, n_dim_state))
#     filtered_state_covariances_combined = np.empty((0, n_duts, n_dim_state, n_dim_state))

#     # Array for storing hits (x, y, z, xerr, yerr, zerr) of possible tracks
#     track_hits_buffer_combined = np.zeros((0, n_duts, n_dim_state))

    track_hits_buffer = 0  # Hits belonging to possbile tracks. Only needed to define variable
    per_event_track_variables_buffer = 0  # Variables like event number, chi2, hit flag
    per_dut_track_variables_buffer = 0  # Variables like charge, number of hits, cluster shape

    n_trees = track_hits.shape[0]  # Number of trees for actual plane
    seed_slope = [0.0, 0.0, 1.0]  # TODO: maybe better guess. Wrong slope could be 'corrected' with larger search radius for in second plane with the disadvantage of having higher multiplicity

    # Arrays needed for Kalman Filter. Results (calculated plane by plane) will be written to these arrays and used for the next plane.
    predicted_states = np.zeros((track_hits.shape[0], n_duts, n_dim_state))
    predicted_state_covariances = np.zeros((track_hits.shape[0], n_duts, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((track_hits.shape[0], n_duts, n_dim_state, n_dim_obs))
    filtered_states = np.zeros_like(predicted_states)
    filtered_state_covariances = np.zeros_like(predicted_state_covariances)
    transition_matrices = np.zeros((track_hits.shape[0], n_duts - 1, n_dim_state, n_dim_state))
    chi2_filter = np.zeros((track_hits.shape[0], n_duts))

    kalman_filter_iteration = 0  # Counting number of Kalman Filter iterations

    for dut_index, dut in enumerate(telescope):
        # Create array for storing possible track hits.
        # Amount of possible tracks is limited by max_multiplicity and number of tree at this plane.
        if dut_index == 0:  # First plane
            # Put seed hits into possible track hits
            track_hits_buffer = track_hits[:, 0, :]
            per_event_track_variables_buffer = np.zeros(shape=(per_event_track_variables.shape[0], track_hits.shape[0]))
            for i in range(per_event_track_variables_buffer.shape[0]):
                per_event_track_variables_buffer[i] = per_event_track_variables[i]
            per_dut_track_variables_buffer = np.zeros(shape=(per_dut_track_variables.shape[0], track_hits.shape[0], track_hits.shape[1]))
            for i in range(per_dut_track_variables_buffer.shape[0]):
                per_dut_track_variables_buffer[i] = per_dut_track_variables[i]
            per_event_track_variables_buffer[2, ~np.isnan(track_hits_buffer[:, 0])] += (1 << dut_index)  # Set hit flag for first plane
            continue  # Continue with next DUT
        else:  # From second plane on
            # Copy results and reset buffer
            possible_track_hits_old = np.copy(track_hits_buffer)
            old_per_event_track_variables = np.copy(per_event_track_variables_buffer)
            old_per_dut_track_variables = np.copy(per_dut_track_variables_buffer)
            track_hits_buffer = np.full(shape=(max_hit_multiplicities[dut_index] * n_trees, dut_index + 1, n_dim_state), fill_value=np.inf)
            per_event_track_variables_buffer = np.full(shape=(old_per_event_track_variables.shape[0], max_hit_multiplicities[dut_index] * n_trees), fill_value=np.inf)
            per_dut_track_variables_buffer = np.full(shape=(old_per_dut_track_variables.shape[0], max_hit_multiplicities[dut_index] * n_trees, dut_index + 1), fill_value=np.inf)
            n_trees = 0  # Reset number of trees after initializing all needed arrays.

            if dut_index == 1:   # Second plane
                # No filter result available for second plane. Thus use seed slope and extrapolate hits from plane before to actual plane.
                line_origins = possible_track_hits_old[:, 0:3]
                line_directions = np.tile(np.array([seed_slope]), (possible_track_hits_old.shape[0], 1))
                fill_kf_results = False  # No Kalman Filter result yet available
                select_duts = [0, 1]
            else:  # From third plane on
                # Use filter result from plane before in order to extrapolate to actual plane.
                line_origins = filtered_states[:, dut_index - 1, :3]
                line_directions = filtered_states[:, dut_index - 1, 3:]
                fill_kf_results = True
                select_duts = [dut_index]
            # Extrapolate hits from plane before to actual plane
            # TODO: check what happens if nan will be extrapolated
            extrapolations = geometry_utils.get_line_intersections_with_dut(
                line_origins=line_origins,
                line_directions=line_directions,
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
# 
#             from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             for i in range(possible_track_hits_old.shape[0]):
#                 try:
#                     ax.scatter(possible_track_hits_old[i, :, 0], possible_track_hits_old[i, :, 1], possible_track_hits_old[i, :, 2], marker='o', label='possible_track_hits_old hits' if i==0 else '')
#                 except:
#                     ax.scatter(possible_track_hits_old[i, 0], possible_track_hits_old[i, 1], possible_track_hits_old[i, 2], marker='o', label='track hits' if i==0 else '')
#             for i in range(track_hits.shape[0]):
#                 ax.scatter(track_hits[i, dut_index, 0], track_hits[i, dut_index, 1], track_hits[i, dut_index, 2], marker='<', label=('actual track hits %i' % dut_index) if i==0 else '')
#             for i in range(filtered_states.shape[0]):
#                 try:
#                     ax.scatter(filtered_states[i, dut_index - 1, 0], filtered_states[i, dut_index - 1, 1], filtered_states[i, dut_index - 1, 2], marker='P', label=('filtered_states %i' % dut_index) if i==0 else '')
#                 except:
#                     pass
#             for i in range(extrapolations[:, 0].shape[0]):
#                 ax.plot([extrapolations[i, 0], line_origins[i, 0]], [extrapolations[i, 1], line_origins[i, 1]], [extrapolations[i, 2], line_origins[i, 2]], ls='--', marker='s', label=('extraploations %i' % dut_index)  if i==0 else '')
#             ax.set_xlabel('X Label')
#             ax.set_ylabel('Y Label')
#             ax.set_zlabel('Z Label')
#             ax.set_title('CKF')
#             plt.legend()
#             plt.show()

            # Search for hits within search radius
            (n_trees, track_hits_buffer, per_event_track_variables_buffer, per_dut_track_variables_buffer,
             predicted_states, predicted_state_covariances, kalman_gains, filtered_states,
             filtered_state_covariances, transition_matrices, chi2_filter) = _search_hits_within_search_radius(
                actual_dut_hits=track_hits[:, dut_index, :],
                actual_per_event_track_variables=per_event_track_variables,
                actual_per_dut_track_variables=per_dut_track_variables,
                possible_track_hits_old=possible_track_hits_old,
                old_per_event_track_variables=old_per_event_track_variables,
                old_per_dut_track_variables=old_per_dut_track_variables,
                track_hits_buffer=track_hits_buffer,
                per_event_track_variables_buffer=per_event_track_variables_buffer,
                per_dut_track_variables_buffer=per_dut_track_variables_buffer,
                n_trees=n_trees, dut_index=dut_index, association_distance=association_distances[dut_index], max_hit_multiplicity=max_hit_multiplicities[dut_index],
                select_hit_duts=select_hit_duts if not len(select_hit_duts) == 0 else [-1], extrapolations=extrapolations,
                predicted_states=predicted_states, predicted_state_covariances=predicted_state_covariances, kalman_gains=kalman_gains,
                filtered_states=filtered_states, filtered_state_covariances=filtered_state_covariances, transition_matrices=transition_matrices,
                chi2=chi2_filter, fill_kf_results=fill_kf_results)

            # Number of possible tracks usually smaller than allowed maximum. Thus, cut not needed possible tracks.
            track_hits_buffer = track_hits_buffer[:n_trees]
            per_event_track_variables_buffer = per_event_track_variables_buffer[:, :n_trees]
            per_dut_track_variables_buffer = per_dut_track_variables_buffer[:, :n_trees, :]
            predicted_states = predicted_states[:n_trees]
            predicted_state_covariances = predicted_state_covariances[:n_trees]
            kalman_gains = kalman_gains[:n_trees]
            filtered_states = filtered_states[:n_trees]
            filtered_state_covariances = filtered_state_covariances[:n_trees]
            transition_matrices = transition_matrices[:n_trees]
            chi2_filter = chi2_filter[:n_trees]
            (predicted_states, predicted_state_covariances, kalman_gains, filtered_states,
             filtered_state_covariances, transition_matrices, chi2_filter) = run_kalman_filter(
                 select_duts=select_duts,
                 use_kalman_filter_result=False if kalman_filter_iteration == 0 else True,  # For first iteration no Kalman Filter available yet. Use it for further iterations
                 track_hits=track_hits_buffer,
                 telescope=telescope,
                 select_fit_duts=select_fit_duts,
                 beam_energy=beam_energy,
                 particle_mass=particle_mass,
                 predicted_states=predicted_states,
                 predicted_state_covariances=predicted_state_covariances,
                 kalman_gains=kalman_gains,
                 filtered_states=filtered_states,
                 filtered_state_covariances=filtered_state_covariances,
                 transition_matrices=transition_matrices,
                 chi2_filter=chi2_filter)
            kalman_filter_iteration += 1  # Increase iteration by one.

            # Calculate for each track a score
            chi2_weight = 0.1  # TODO: adjust this? Put this as parameter
            missing_hits = np.count_nonzero(np.isnan(track_hits_buffer[:, np.array([can_have_hits])[np.array([can_have_hits]) < track_hits_buffer.shape[1]], 0]), axis=1)
            track_score = 2 * track_hits_buffer.shape[1] - missing_hits - chi2_weight * np.nansum(chi2_filter, axis=1)
#             plt.hist(track_score, bins=np.arange(0, 20, 0.5))
#             plt.show()

            actual_chi2 = chi2_filter[:, dut_index]
            sel = ~np.isnan(actual_chi2)
            actual_chi2 = actual_chi2[sel]
            max_delta_chi = [50, 50, 50, 50, 50, 50, 50, 10]
            # print np.percentile(actual_chi2, q=68.27), np.percentile(actual_chi2, q=90.0), np.percentile(actual_chi2, q=95.45), np.percentile(actual_chi2, q=99.73), dut_index
#             plt.hist(actual_chi2, bins=np.arange(0, 150, 0.5), label='Filter plane %i' % dut_index)
#             plt.legend()
#             plt.show()

            _, seed_id_indices = np.unique(per_event_track_variables_buffer[3], return_index=True)
            # Do quality selection in order to reduce possible combinatorics. Actual DUT is excluded in order to not bias it.
            no_quality_duts = set(range(len(telescope))) - set(can_have_hits) - set(select_hit_duts)
            if dut_index not in no_quality_duts:
                good_track_selection = _check_for_good_hits(
                    hits=track_hits_buffer[:, :, 0:3],
                    actual_track_chi2=chi2_filter[:, dut_index],
                    track_score=track_score,
                    missing_hits=missing_hits,
                    actual_dut=dut_index,
                    # Take 1 sigma as max chi2. Dynamic limit is useful here since chi2 from Kalman Filter will get better with each plane.
                    max_delta_chi2=max_delta_chi[dut_index], #np.percentile(actual_chi2, q=qs[dut_index]),  # max_delta_chi2
                    max_delta_score=3,  # TODO: check this
                    seed_ids=per_event_track_variables_buffer[3],
                    indices=seed_id_indices,
                    max_missing_hits=max_missing_hits)
                # Apply quality selection
                track_hits_buffer = track_hits_buffer[good_track_selection]
                per_event_track_variables_buffer = per_event_track_variables_buffer[:, good_track_selection]
                per_dut_track_variables_buffer = per_dut_track_variables_buffer[:, good_track_selection, :]
                predicted_states = predicted_states[good_track_selection]
                predicted_state_covariances = predicted_state_covariances[good_track_selection]
                kalman_gains = kalman_gains[good_track_selection]
                filtered_states = filtered_states[good_track_selection]
                filtered_state_covariances = filtered_state_covariances[good_track_selection]
                transition_matrices = transition_matrices[good_track_selection]
                chi2_filter = chi2_filter[good_track_selection]
                track_score = track_score[good_track_selection]

#             # Plotting for debugging
#             from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             for i in range(track_hits_buffer.shape[0]):
#                 try:
#                     ax.scatter(track_hits_buffer[i, :, 0], track_hits_buffer[i, :, 1], track_hits_buffer[i, :, 2], marker='o', label='track hits' if i==0 else '')
#                 except:
#                     ax.scatter(track_hits_buffer[i, 0], track_hits_buffer[i, 1], track_hits_buffer[i, 2], marker='o', label='track hits' if i==0 else '')
#             for i in range(track_hits.shape[0]):
#                 ax.scatter(track_hits[i, dut_index, 0], track_hits[i, dut_index, 1], track_hits[i, dut_index, 2], marker='<', label=('actual track hits %i' % dut_index) if i==0 else '')
#             for i in range(filtered_states.shape[0]):
#                 try:
#                     ax.scatter(filtered_states[i, dut_index - 1, 0], filtered_states[i, dut_index - 1, 1], filtered_states[i, dut_index - 1, 2], marker='P', label=('filtered_states %i' % dut_index) if i==0 else '')
#                 except:
#                     pass
#             for i in range(extrapolations[:, 0].shape[0]):
#                 ax.plot([extrapolations[i, 0], line_origins[i, 0]], [extrapolations[i, 1], line_origins[i, 1]], [extrapolations[i, 2], line_origins[i, 2]], ls='--', marker='s', label=('extraploations %i' % dut_index)  if i==0 else '')
#             ax.set_xlabel('X Label')
#             ax.set_ylabel('Y Label')
#             ax.set_zlabel('Z Label')
#             ax.set_title('CKF')
#             plt.legend()
#             plt.show()

        # If last DUT is reached, store Kalman Filter results to array in order to use them for the final Kalman Smoother
        if dut_index == n_duts - 1:
            predicted_states_all_planes = predicted_states
            predicted_state_covariances_all_planes = predicted_state_covariances
            filtered_states_all_planes = filtered_states
            filtered_state_covariances_all_planes = filtered_state_covariances
            transition_matrices_all_planes = transition_matrices
            track_hits_all_planes = track_hits_buffer
            per_event_track_variables = per_event_track_variables_buffer
            per_dut_track_variables = per_dut_track_variables_buffer

    # Run Kalman Smoother
    smoothed_states, _, chi2_smooth = run_kalman_smoother(
        track_hits=track_hits_all_planes,
        telescope=telescope,
        select_fit_duts=select_fit_duts,
        beam_energy=beam_energy,
        particle_mass=particle_mass,
        filtered_states=filtered_states_all_planes,
        filtered_state_covariances=filtered_state_covariances_all_planes,
        predicted_states=predicted_states_all_planes,
        predicted_state_covariances=predicted_state_covariances_all_planes,
        transition_matrices=transition_matrices_all_planes)

    # Sum up chi2 for all planes and divide by number of hits
    # chi2_normalized = np.nansum(chi2_smooth, axis=1) / (3 * (np.count_nonzero(~np.isnan(chi2_smooth[:, select_fit_duts]), axis=1) - 3))
    chi2_normalized = np.nansum(chi2_smooth[:, select_fit_duts], axis=1) / (3 * (np.count_nonzero(~np.isnan(chi2_smooth[:, select_fit_duts]), axis=1) - 3))
    per_event_track_variables[1] = chi2_normalized

    # Sort chi2 (per event). Have to make sure that only within same event chi2 is sorted.
    _, unique_event_number_indices, cnt = np.unique(per_event_track_variables[0], return_index=True, return_counts=True)
    sorted_indices = []
    for i in range(len(unique_event_number_indices)):
        if i == len(unique_event_number_indices) - 1:
            sorted_indices.append(list(np.argsort(chi2_normalized[unique_event_number_indices[i]:])))
        else:
            sorted_indices.append(list(np.argsort(chi2_normalized[unique_event_number_indices[i]:unique_event_number_indices[i + 1]])))
    offset = np.repeat(unique_event_number_indices, cnt)
    sorted_indices = list(itertools.chain.from_iterable(sorted_indices))
    sorted_indices = np.array(sorted_indices)
    sorted_indices += offset

    # Sort tracks
    track_hits_all_planes = track_hits_all_planes[sorted_indices, :, :]
    per_event_track_variables = per_event_track_variables[:, sorted_indices]
    smoothed_states = smoothed_states[sorted_indices, :, :]
    # Now can put chi2 to array
    per_event_track_variables[1] = chi2_normalized[sorted_indices]
    per_dut_track_variables = per_dut_track_variables[:, sorted_indices, :]

    # Select only tracks where hit is only used once (based on lowest chi2)
    final_track_hits, final_per_event_track_variables, final_smoothed_states, final_per_dut_track_variables = _remove_duplicated_hits(
        track_candidates=track_hits_all_planes,
        track_estimates=smoothed_states,
        per_event_track_variables=per_event_track_variables,
        per_dut_track_variables=per_dut_track_variables)

    # Select for every seed the best track candidate (track score/chi2)
#     selection = _final_track_selection(track_chi2=per_event_track_variables[1],
#                                        seed_id=per_event_track_variables[3])
# 
#     final_track_hits = track_hits_all_planes[selection, :, :]
#     final_smoothed_states = smoothed_states[selection, :, :]
#     final_per_event_track_variables = per_event_track_variables[:, selection]
#     final_per_dut_track_variables = per_dut_track_variables[:, selection, :]

#     final_track_hits = track_hits_buffer_combined
#     final_smoothed_states = smoothed_states
#     final_per_event_track_variables = per_event_track_variables_buffer_combined
#     final_per_dut_track_variables = per_dut_track_variables_buffer_combined

    return final_track_hits, final_smoothed_states, final_per_event_track_variables, final_per_dut_track_variables


def run_kalman_filter(select_duts, use_kalman_filter_result, track_hits, telescope, select_fit_duts, beam_energy, particle_mass,
                      predicted_states=None, predicted_state_covariances=None, kalman_gains=None,
                      filtered_states=None, filtered_state_covariances=None, transition_matrices=None, chi2_filter=None):

    # Initialize all needed matrices for Kalman Filter
    ckf = prepare_ckf(
        track_hits=track_hits,
        telescope=telescope,
        select_fit_duts=select_fit_duts,
        beam_energy=beam_energy,
        particle_mass=particle_mass)

    # Run Kalman Filter
    return ckf.filter_iterative(
        select_duts=select_duts,
        use_kalman_filter_result=use_kalman_filter_result,
        observations=track_hits[:, :, 0:3],
        predicted_states=predicted_states,
        predicted_state_covariances=predicted_state_covariances,
        kalman_gains=kalman_gains,
        filtered_states=filtered_states,
        filtered_state_covariances=filtered_state_covariances,
        transition_matrices_update=transition_matrices,
        chi2=chi2_filter)


def run_kalman_smoother(track_hits, telescope, select_fit_duts, beam_energy, particle_mass, filtered_states, filtered_state_covariances, predicted_states, predicted_state_covariances, transition_matrices):
    # Initialize all needed matrices for Kalman Filter
    ckf = prepare_ckf(
        track_hits=track_hits,
        telescope=telescope,
        select_fit_duts=select_fit_duts,
        beam_energy=beam_energy,
        particle_mass=particle_mass)

    # store needed prediction and filter results at last DUT in order to use them for final smoothing
    smoothed_states, smoothed_state_covariances, chi2_smooth = ckf.smooth(
        filtered_states=filtered_states,
        filtered_state_covariances=filtered_state_covariances,
        predicted_states=predicted_states,
        predicted_state_covariances=predicted_state_covariances,
        transition_matrices=transition_matrices,
        observations=track_hits[:, :, 0:3])

    # Check for invalid values (NaN)
    if np.any(np.isnan(smoothed_states)):
        logging.warning('Smoothed state estimates contain invalid values (NaNs). Check input of Kalman Filter.')

    return smoothed_states, smoothed_state_covariances, chi2_smooth


def prepare_ckf(track_hits, telescope, select_fit_duts, beam_energy, particle_mass):
    alignment = []
    material_budget = []
    all_dut_planes = [dut for dut in telescope]
    for dut in all_dut_planes:
        alignment.append([dut.translation_x, dut.translation_y, dut.translation_z, dut.rotation_alpha, dut.rotation_beta, dut.rotation_gamma])
        # TODO: take rotation into account for material budget
        material_budget.append(dut.material_budget)
    alignment = np.array(alignment)
    material_budget = np.array(material_budget)

    chunk_size = track_hits.shape[0]
    n_duts = track_hits.shape[1]

    # Calculate multiple scattering
    momentum = np.sqrt(beam_energy**2 - particle_mass**2)
    beta = momentum / beam_energy  # almost 1

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in all_dut_planes:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)

    if np.any(np.isclose(material_budget[z_sorted_dut_indices[:-1]], 0.0)):
        raise ValueError("Material budget is zero.")

    # rms angle of multiple scattering
    thetas = np.array(((13.6 / momentum / beta) * np.sqrt(material_budget) * (1. + 0.038 * np.log(material_budget))))
    # error on z-position
    z_error = 1e3  # Assume 1 mm

    # express transition and observation offset matrices
    # these are additional offsets, which are not used at the moment
    transition_offsets = np.zeros((chunk_size, n_duts - 1, 6), dtype=np.float64)
    observation_offsets = np.zeros((chunk_size, n_duts, 3), dtype=np.float64)

    # express initial state. Contains (x_pos, y_pos, z_pos, slope_x, slope_y, slope_z).
    initial_state_mean = np.zeros((chunk_size, 6), dtype=np.float64)

    # express observation matrix, only observe (x,y,z)
    observation_matrices = np.zeros((chunk_size, n_duts, 3, 6), dtype=np.float64)
    observation_matrices[:, :, 0, 0] = 1.0
    observation_matrices[:, :, 1, 1] = 1.0
    observation_matrices[:, :, 2, 2] = 1.0
    # express observation covariance matrices
    observation_covariances = np.zeros((chunk_size, n_duts, 3, 3), dtype=np.float64)

    # express initial state covariance matrices
    initial_state_covariance = np.zeros((chunk_size, 6, 6), dtype=np.float64)
    # error on initial slope is roughly divergence of beam (5 mrad).
    initial_state_covariance[:, 3, 3] = np.square(5e-3)
    initial_state_covariance[:, 4, 4] = np.square(5e-3)
    initial_state_covariance[:, 5, 5] = np.square(5e-3)

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        # Take cluster hit position error as measurement error for duts which have a hit.
        # For those who have no hit, need no error, since the should not be included in fit via fit selection
        duts_with_hits = np.array(range(n_duts), dtype=np.int)[~np.isnan(actual_hits[:, 0])]
        observation_covariances[index, duts_with_hits, 0, 0] = np.square(actual_hits[duts_with_hits, 3])
        observation_covariances[index, duts_with_hits, 1, 1] = np.square(actual_hits[duts_with_hits, 4])
        observation_covariances[index, duts_with_hits, 2, 2] = np.square(z_error)
        if np.isnan(actual_hits[z_sorted_dut_indices[0], 0]):  # The first plane has no hit
            # Take planes from fit selction and fit a line to the hits,
            # then extrapolate the line to first plane in order to find initial state.
            # The position error is estimated with the pixel size.
            # TODO: Can't we handle this as any other scattering plane with error=0?
            # Edit: Any plane without hit is treated as scatter plane.
            try:
                # Fit all DUTs with hits
                offset, slope = line_fit_3d(positions=actual_hits[select_fit_duts, :3])
                # TODO: For lower energies and lighter particles use the first hit DUT as position for the first scatter plane
                # Fit the first 2 DUTs with hits
                # offset, slope = line_fit_3d(positions=actual_hits[z_sorted_fit_dut_indices, :3], n=2)
            except np.linalg.linalg.LinAlgError:
                offset, slope = np.nan, np.nan

            intersections = geometry_utils.get_line_intersections_with_dut(
                line_origins=np.array([offset]),
                line_directions=np.array([slope]),
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [intersections[0, 0], intersections[0, 1], intersections[0, 2], 0.0, 0.0, 1.0]
            initial_state_covariance[index, 0, 0] = 0.0
            initial_state_covariance[index, 1, 1] = 0.0
            # initial_state_covariance[index, 0, 0] = np.square(first_dut_pixel_size[0])
            # initial_state_covariance[index, 1, 1] = np.square(first_dut_pixel_size[1])
        else:  # The first plane has a hit
            # If first plane should be included in track building, take first dut hit as initial value and
            # its corresponding cluster position error as the error on the measurement.
            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [actual_hits[z_sorted_dut_indices[0], 0], actual_hits[z_sorted_dut_indices[0], 1], actual_hits[z_sorted_dut_indices[0], 2], 0.0, 0.0, 1.0]
            initial_state_covariance[index, 0, 0] = np.square(actual_hits[z_sorted_dut_indices[0], 3])  # x_err
            initial_state_covariance[index, 1, 1] = np.square(actual_hits[z_sorted_dut_indices[0], 4])  # y_err
            initial_state_covariance[index, 2, 2] = np.square(z_error)

    # Init CKF
    ckf = kalman.CombinatorialKalmanFilter(
        dut_planes=all_dut_planes,
        select_fit_duts=select_fit_duts,
        thetas=thetas[:n_duts],
        z_sorted_dut_indices=z_sorted_dut_indices,
        transition_offsets=transition_offsets,
        observation_matrices=observation_matrices,
        observation_covariances=observation_covariances,
        observation_offsets=observation_offsets,
        initial_state_means=initial_state_mean,
        initial_state_covariances=initial_state_covariance)

    return ckf


@njit
def _search_hits_within_search_radius(actual_dut_hits, actual_per_event_track_variables, actual_per_dut_track_variables, possible_track_hits_old, old_per_event_track_variables, 
                                      old_per_dut_track_variables, track_hits_buffer, per_event_track_variables_buffer, per_dut_track_variables_buffer,
                                      n_trees, dut_index, association_distance, max_hit_multiplicity,
                                      select_hit_duts, extrapolations, predicted_states, predicted_state_covariances,
                                      kalman_gains, filtered_states, filtered_state_covariances, transition_matrices, chi2, fill_kf_results=False):
    possible_track_hit_index = 0
    # Arrays in order to store KF results of all found trees
    predicted_states_new = np.zeros(shape=(track_hits_buffer.shape[0], predicted_states.shape[1], predicted_states.shape[2]))
    predicted_state_covariances_new = np.zeros(shape=(track_hits_buffer.shape[0], predicted_state_covariances.shape[1],
                                                      predicted_state_covariances.shape[2], predicted_state_covariances.shape[3]))
    kalman_gains_new = np.zeros(shape=(track_hits_buffer.shape[0], kalman_gains.shape[1], kalman_gains.shape[2], kalman_gains.shape[3]))
    filtered_states_new = np.zeros_like(predicted_states_new)
    filtered_state_covariances_new = np.zeros_like(predicted_state_covariances_new)
    transition_matrices_new = np.zeros(shape=(track_hits_buffer.shape[0], transition_matrices.shape[1],
                                              transition_matrices.shape[2], transition_matrices.shape[3]))
    chi2_new = np.zeros(shape=(track_hits_buffer.shape[0], chi2.shape[1]))
    actual_event_number_extrapolation = old_per_event_track_variables[0, 0]

    new_start_index = 0  # index where to search for new hits within search radius (optimized in order to start not for every event number at the beginning)
    last_hit_index = 0  # index of last event number change (will be used as new start index)
    start_index_is_updated = False  # variable in order to indicate if start_index is already updated. Must be only updated at beginning of one event.

    extrapolations_x, extrapolations_y = extrapolations[:, 0], extrapolations[:, 1]
    # Search for hits within search radius and fill possible tracks array
    for tree_index in range(extrapolations_x.shape[0]):  # Loop over all trees
        actual_extrapolation_x = extrapolations_x[tree_index]
        actual_extrapolation_y = extrapolations_y[tree_index]
        n_multiplicity_per_tree = 0
        found_hit_for_actual_extrapolation = False
        old_event_number_extrapolation = actual_event_number_extrapolation
        actual_event_number_extrapolation = old_per_event_track_variables[0, tree_index]  # Event number of actual extrapolation

        for hit_index in range(actual_dut_hits.shape[0]):  # Loop over hits in actual plane and check for matching hits.
            if tree_index > 0:
                if actual_event_number_extrapolation != old_event_number_extrapolation:  # Set new start index if all extrapolations of one event number are done.
                    new_start_index = last_hit_index
                    start_index_is_updated = False

            if hit_index + new_start_index >= actual_dut_hits.shape[0] - 1:  # Go to next extrapolation if reached end of track hits
                break
            possible_hit = actual_dut_hits[hit_index + new_start_index]
            actual_event_number_search_hit = actual_per_event_track_variables[0, hit_index + new_start_index]

            # Check if hit is still within actual event
            if actual_event_number_extrapolation < actual_event_number_search_hit:
                # Reached all possible hits of same event. Thus go next extrapolation and search again. Store hit index, in order to know where new event starts.
                if not start_index_is_updated:
                    last_hit_index = last_hit_index + hit_index
                    start_index_is_updated = True
                break
            elif actual_event_number_extrapolation > actual_event_number_search_hit:
                continue
            else:
                # Check if actual hit is within search radius
                if (np.abs(actual_extrapolation_x - possible_hit[0]) <= association_distance[0]) and (np.abs(actual_extrapolation_y - possible_hit[1]) <= association_distance[1]):
                    if n_multiplicity_per_tree >= max_hit_multiplicity:
                        # TODO: sort hits according to residuum and take always the best max_hit_multiplicity hits
                        # Found maximum number of allowed hits within search radius for actual extrapolation. Thus go to next extrapolation
                        print('WARNING: max multiplicity reached!')
                        break
#                         same_tree_selection = (per_event_track_variables_buffer[3, :] == tree_index)
#                         actual_track_hits = track_hits_buffer[same_tree_selection, dut_index]
#                         residuals_x = np.abs(actual_extrapolation_x - actual_track_hits[:, 0])
#                         residuals_y = np.abs(actual_extrapolation_y - actual_track_hits[:, 1])
#                         distances = np.sqrt(residuals_x**2 + residuals_y**2)
#                         largest_distance = np.max(distances)
#                         actual_residual_x = np.abs(actual_extrapolation_x - possible_hit[0])
#                         actual_residual_y = np.abs(actual_extrapolation_y - possible_hit[1])
#                         actual_distance = np.sqrt(actual_residual_x**2 + actual_residual_y**2)
#                         if actual_distance >= largest_distance:  # only break if actual distance larger than actual distance
#                             break
                    # Fill hits. TODO: maybe put this into _fill function
                    track_hits_buffer[possible_track_hit_index, :dut_index] = possible_track_hits_old[tree_index]  # Hits from planes before
                    track_hits_buffer[possible_track_hit_index, dut_index] = possible_hit  # Fill actual hit within search radius
                    for i in range(per_event_track_variables_buffer.shape[0]):
                        per_event_track_variables_buffer[i, possible_track_hit_index] = old_per_event_track_variables[i, tree_index]
                    per_event_track_variables_buffer[2, possible_track_hit_index] += (1 << dut_index)  # Set hit flag. Note: cannot use logical or since track variables contain floats like chi2.
                    if not fill_kf_results:  # set seed ID only for first plane. For other planes, copy from array before.
                        per_event_track_variables_buffer[3, possible_track_hit_index] = tree_index  # Set seed ID
                    for i in range(per_dut_track_variables_buffer.shape[0]):
                        per_dut_track_variables_buffer[i, possible_track_hit_index, :dut_index] = old_per_dut_track_variables[i, tree_index, :dut_index]
                        per_dut_track_variables_buffer[i, possible_track_hit_index, dut_index] = actual_per_dut_track_variables[i, hit_index + new_start_index, dut_index]
                    if fill_kf_results:  # Fill results from kalman filter for found tree
                        predicted_states_new[possible_track_hit_index, :dut_index] = predicted_states[tree_index, :dut_index]
                        predicted_state_covariances_new[possible_track_hit_index, :dut_index] = predicted_state_covariances[tree_index, :dut_index]
                        kalman_gains_new[possible_track_hit_index, :dut_index] = kalman_gains[tree_index, :dut_index]
                        filtered_states_new[possible_track_hit_index, :dut_index] = filtered_states[tree_index, :dut_index]
                        filtered_state_covariances_new[possible_track_hit_index, :dut_index] = filtered_state_covariances[tree_index, :dut_index]
                        transition_matrices_new[possible_track_hit_index, :dut_index] = transition_matrices[tree_index, :dut_index]
                        chi2_new[possible_track_hit_index, :dut_index] = chi2[tree_index, :dut_index]
                    possible_track_hit_index += 1
                    n_multiplicity_per_tree += 1
                    found_hit_for_actual_extrapolation = True  # found at least one hit within search radius

        # Check if hit is required for this DUT. If no hit is required for this DUT and no hit was found, append np.nan to possible tracks.
        # TODO: allow also always no dut hit for each tree

        # Add missing hit (np.nan) if no hit was required (dut not in hit dut selection) or found hit but no was required (allow for every hit also no hit)
        if (not found_hit_for_actual_extrapolation and dut_index not in select_hit_duts):# or (found_hit_for_actual_extrapolation and dut_index not in select_hit_duts):
            track_hits_buffer[possible_track_hit_index, :dut_index] = possible_track_hits_old[tree_index]  # Hits from planes before
            track_hits_buffer[possible_track_hit_index, dut_index] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]  # Fill actual hit within search radius
            for i in range(per_event_track_variables_buffer.shape[0]):
                per_event_track_variables_buffer[i, possible_track_hit_index] = old_per_event_track_variables[i, tree_index]
            if not fill_kf_results:  # set seed ID only for first plane. For other planes, copy from array before.
                per_event_track_variables_buffer[3, possible_track_hit_index] = tree_index  # Set seed ID
            for i in range(per_dut_track_variables_buffer.shape[0]):
                per_dut_track_variables_buffer[i, possible_track_hit_index, :dut_index] = old_per_dut_track_variables[i, tree_index, :dut_index]
            per_dut_track_variables_buffer[0, possible_track_hit_index, dut_index] = np.nan  # Set charge to NaN
            per_dut_track_variables_buffer[1, possible_track_hit_index, dut_index] = 0  # Set n hits to 0
            per_dut_track_variables_buffer[2, possible_track_hit_index, dut_index] = 0  # Set cluster shape to NaN
            per_dut_track_variables_buffer[3, possible_track_hit_index, dut_index] = 0  # Set n cluster to NaN

            if fill_kf_results:  # Fill results from kalman filter for found tree
                predicted_states_new[possible_track_hit_index, :dut_index] = predicted_states[tree_index, :dut_index]
                predicted_state_covariances_new[possible_track_hit_index, :dut_index] = predicted_state_covariances[tree_index, :dut_index]
                kalman_gains_new[possible_track_hit_index, :dut_index] = kalman_gains[tree_index, :dut_index]
                filtered_states_new[possible_track_hit_index, :dut_index] = filtered_states[tree_index, :dut_index]
                filtered_state_covariances_new[possible_track_hit_index, :dut_index] = filtered_state_covariances[tree_index, :dut_index]
                transition_matrices_new[possible_track_hit_index, :dut_index] = transition_matrices[tree_index, :dut_index]
                chi2_new[possible_track_hit_index, :dut_index] = chi2[tree_index, :dut_index]
            n_multiplicity_per_tree += 1  # Did not found a hit, but tree is not dead, thus count one up
            possible_track_hit_index += 1

        n_trees += n_multiplicity_per_tree

    return (n_trees, track_hits_buffer, per_event_track_variables_buffer, per_dut_track_variables_buffer,
            predicted_states_new, predicted_state_covariances_new, kalman_gains_new,
            filtered_states_new, filtered_state_covariances_new, transition_matrices_new, chi2_new)


@njit
def _check_for_good_hits(hits, actual_dut, actual_track_chi2, track_score, missing_hits, max_delta_chi2, max_delta_score, seed_ids, indices, max_missing_hits):
    good_track_selection = np.ones(shape=(hits.shape[0]), dtype=np.bool_)
    actual_hits = hits[:, actual_dut]

    seed_index = 0
    actual_seed_id = seed_ids[0]
    highest_track_score = np.max(track_score[indices[seed_index]:indices[seed_index + 1]])
    for index in range(actual_hits.shape[0]):
        old_seed_id = actual_seed_id
        actual_seed_id = seed_ids[index]
        if old_seed_id != actual_seed_id:
            seed_index += 1
            # Update highest track score if seed has changed
            if seed_index >= indices.shape[0] - 1:
                highest_track_score = np.max(track_score[indices[seed_index]:])
            else:
                highest_track_score = np.max(track_score[indices[seed_index]:indices[seed_index + 1]])

        # Check for total number of missing hits
        if missing_hits[index] > max_missing_hits:
            good_track_selection[index] = 0
        # check for chi2 contribution
        if ~np.isnan(actual_track_chi2[index]):
            if actual_track_chi2[index] > max_delta_chi2:
                # print hits[index, :, 0], actual_track_chi2[index], actual_dut, 'chi2'
                good_track_selection[index] = 0
        # check track score
        if (highest_track_score - track_score[index]) > max_delta_score:
            # print hits[index, :, 0], actual_track_chi2[index], highest_track_score, track_score[index], 'score'
            good_track_selection[index] = 0

    return good_track_selection


@njit
def _remove_duplicated_hits(track_candidates, track_estimates, per_event_track_variables, per_dut_track_variables, max_tracks_per_event=40):
    n_duts = track_candidates.shape[1]
    n_dim_state = track_candidates.shape[2]
    final_tracks_buffer = np.zeros((max_tracks_per_event, n_duts, n_dim_state))  # Array in order to store final tracks per event
    per_event_track_variables_buffer = np.zeros((per_event_track_variables.shape[0], max_tracks_per_event))  # Array in order to store final tracks per event
    track_estimates_buffer = np.zeros((max_tracks_per_event, n_duts, n_dim_state))
    per_dut_track_variables_buffer = np.zeros((per_dut_track_variables.shape[0], max_tracks_per_event, n_duts))  # Array in order to store final tracks per event

    init = True
    found_first_track = False
    final_tracks_buffer_index = 0
    actual_event_number = per_event_track_variables[0, 0]
    for track_index in range(track_candidates.shape[0]):  # Loop over hits
        old_event_number = actual_event_number
        actual_event_number = per_event_track_variables[0, track_index]

        if actual_event_number != old_event_number:  # Have checked all hits of same event
            # Set number of tracks per event
            per_event_track_variables_buffer[3, :] = final_tracks_buffer_index + 1
            # Write found tracks to result array
            if init:
                final_tracks = final_tracks_buffer[:final_tracks_buffer_index]
                final_per_event_track_variables = per_event_track_variables_buffer[:, :final_tracks_buffer_index]
                final_track_estimates = track_estimates_buffer[:final_tracks_buffer_index]
                final_per_dut_track_variables = per_dut_track_variables_buffer[:, :final_tracks_buffer_index, :]
                init = False
            else:
                final_tracks = np.concatenate((final_tracks, final_tracks_buffer[:final_tracks_buffer_index]), axis=0)
                final_per_event_track_variables = np.concatenate((final_per_event_track_variables, per_event_track_variables_buffer[:, :final_tracks_buffer_index]), axis=1)
                final_track_estimates = np.concatenate((final_track_estimates, track_estimates_buffer[:final_tracks_buffer_index]), axis=0)
                final_per_dut_track_variables = np.concatenate((final_per_dut_track_variables, per_dut_track_variables_buffer[:, :final_tracks_buffer_index, :]), axis=1)
            # Reset used tracks per event
            final_tracks_buffer = np.zeros((max_tracks_per_event, n_duts, n_dim_state))
            per_event_track_variables_buffer = np.zeros((per_event_track_variables.shape[0], max_tracks_per_event))  # Array in order to store final tracks per event
            track_estimates_buffer = np.zeros((max_tracks_per_event, n_duts, n_dim_state))
            per_dut_track_variables_buffer = np.zeros((per_dut_track_variables.shape[0], max_tracks_per_event, n_duts))
            final_tracks_buffer_index = 0
            found_first_track = False

        if not found_first_track:
            final_tracks_buffer[final_tracks_buffer_index] = track_candidates[track_index]
            track_estimates_buffer[final_tracks_buffer_index] = track_estimates[track_index]
            for i in range(per_event_track_variables_buffer.shape[0]):
                per_event_track_variables_buffer[i, final_tracks_buffer_index] = per_event_track_variables[i, track_index]
            for i in range(per_dut_track_variables_buffer.shape[0]):
                per_dut_track_variables_buffer[i, final_tracks_buffer_index] = per_dut_track_variables[i, track_index]
            final_tracks_buffer_index += 1
            found_first_track = True
        else:  # Check if actual track has already used hit
            has_used_hit = False
            for dut_index in range(n_duts):  # Loop over all planes
                actual_x = track_candidates[track_index, dut_index, 0]
                actual_y = track_candidates[track_index, dut_index, 1]
                if actual_x in list(final_tracks_buffer[:, dut_index, 0]) and actual_y in list(final_tracks_buffer[:, dut_index, 1]):  # Track has already used hit
                    has_used_hit = True
                    break  # Exit loop in order to continue with next tracks since found alreay a used hit
                else:  # Check next plane
                    continue

            if not has_used_hit:  # No hit of actual track was used before. Thus write it to final track array
                if final_tracks_buffer_index >= max_tracks_per_event - 1:
                    break
                else:
                    final_tracks_buffer[final_tracks_buffer_index] = track_candidates[track_index]
                    track_estimates_buffer[final_tracks_buffer_index] = track_estimates[track_index]
                    for i in range(per_event_track_variables_buffer.shape[0]):
                        per_event_track_variables_buffer[i, final_tracks_buffer_index] = per_event_track_variables[i, track_index]
                    for i in range(per_dut_track_variables_buffer.shape[0]):
                        per_dut_track_variables_buffer[i, final_tracks_buffer_index] = per_dut_track_variables[i, track_index]
                    final_tracks_buffer_index += 1

    return final_tracks, final_per_event_track_variables, final_track_estimates, final_per_dut_track_variables


@njit
def _final_track_selection(track_chi2, seed_id):
    selection = np.zeros(shape=track_chi2.shape[0], dtype=np.bool_)

    actual_seed_id = seed_id[0]
    for index in range(track_chi2.shape[0]):
        old_seed_id = actual_seed_id
        actual_seed_id = seed_id[index]
        actual_track_chi2 = track_chi2[index]

        if index == 0 or actual_seed_id != old_seed_id:  # set first track candidate as best candidate for very first track or if new seed id
            best_track_chi2 = track_chi2[index]
            best_track_index = index
            selection[index] = 1  # set new best track
            continue

        if np.abs(actual_track_chi2 - best_track_chi2) < 1:  # Unset best track candidate if chi2 too close
            selection[best_track_index] = 0  # Unset best track candidate
            best_track_index = index
            best_track_chi2 = track_chi2[index]

        if actual_track_chi2 < best_track_chi2:
            selection[index] = 1  # set new best track
            selection[best_track_index] = 0  # remove old track
            best_track_index = index
            best_track_chi2 = track_chi2[index]

    return selection
