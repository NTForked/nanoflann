/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-2026 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "test_helpers.h"

TEST(kdtree, L1_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int knn = 1; knn < 20; knn += 3)
    {
        for (int i = 0; i < 500; i++)
        {
            L1_vs_bruteforce_test<float>(10, 2, knn);

            L1_vs_bruteforce_test<float>(100, 2, knn);
            L1_vs_bruteforce_test<float>(100, 3, knn);
            L1_vs_bruteforce_test<float>(100, 7, knn);

            L1_vs_bruteforce_test<double>(100, 2, knn);
            L1_vs_bruteforce_test<double>(100, 3, knn);
            L1_vs_bruteforce_test<double>(100, 7, knn);
        }
    }
}

TEST(kdtree, L1_vs_bruteforce_rknn)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (size_t knn = 1; knn < 20; knn += 3)
    {
        for (int r = 1; r < 5; r++)
        {
            const float rf = static_cast<float>(r);
            for (int i = 0; i < 100; i++)
            {
                rknn_L1_vs_bruteforce_test<float>(100, 2, knn, 9.0f * rf * rf);
                rknn_L1_vs_bruteforce_test<float>(100, 3, knn, 9.0f * rf * rf);
                rknn_L1_vs_bruteforce_test<float>(100, 7, knn, 9.0f * rf * rf);

                rknn_L1_vs_bruteforce_test<double>(100, 2, knn, 9.0 * r * r);
                rknn_L1_vs_bruteforce_test<double>(100, 3, knn, 9.0 * r * r);
                rknn_L1_vs_bruteforce_test<double>(100, 7, knn, 9.0 * r * r);
            }
        }
    }
}

TEST(kdtree, L2_vs_L2_simple)
{
    for (int nResults = 1; nResults < 10; nResults++)
    {
        L2_vs_L2_simple_test<float>(5, nResults);

        L2_vs_L2_simple_test<float>(100, nResults);
        L2_vs_L2_simple_test<double>(100, nResults);
    }
}

TEST(kdtree, robust_empty_tree)
{
    // Try to build a tree with 0 data points, to test
    // robustness against this situation:
    PointCloud<double> cloud;

    double query_pt[3] = {0.5, 0.5, 0.5};

    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3 /* dim */
        >
        my_kd_tree_simple_t;

    my_kd_tree_simple_t index1(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    // Now we will try to search in the tree, and WE EXPECT a result of
    // no neighbors found if the error detection works fine:
    const size_t                    num_results = 1;
    std::vector<size_t>             ret_index(num_results);
    std::vector<double>             out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
    bool result = index1.findNeighbors(resultSet, &query_pt[0]);
    EXPECT_EQ(result, false);
}

TEST(kdtree, L2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int knn = 1; knn < 20; knn += 3)
    {
        for (int i = 0; i < 500; i++)
        {
            L2_vs_bruteforce_test<float>(10, 2, knn);

            L2_vs_bruteforce_test<float>(100, 2, knn);
            L2_vs_bruteforce_test<float>(100, 3, knn);
            L2_vs_bruteforce_test<float>(100, 7, knn);

            L2_vs_bruteforce_test<double>(100, 2, knn);
            L2_vs_bruteforce_test<double>(100, 3, knn);
            L2_vs_bruteforce_test<double>(100, 7, knn);
        }
    }
}

TEST(kdtree, box_L2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 500; i++)
    {
        box_L2_vs_bruteforce_test<float>(10, 2);

        box_L2_vs_bruteforce_test<float>(100, 2);
        box_L2_vs_bruteforce_test<float>(100, 3);
        box_L2_vs_bruteforce_test<float>(100, 7);

        box_L2_vs_bruteforce_test<double>(100, 2);
        box_L2_vs_bruteforce_test<double>(100, 3);
        box_L2_vs_bruteforce_test<double>(100, 7);
    }
}

TEST(kdtree, L2_vs_bruteforce_rknn)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (size_t knn = 1; knn < 20; knn += 3)
    {
        for (int r = 1; r < 5; r++)
        {
            const float rf = static_cast<float>(r);
            for (int i = 0; i < 100; i++)
            {
                rknn_L2_vs_bruteforce_test<float>(100, 2, knn, 9.0f * rf * rf);
                rknn_L2_vs_bruteforce_test<float>(100, 3, knn, 9.0f * rf * rf);
                rknn_L2_vs_bruteforce_test<float>(100, 7, knn, 9.0f * rf * rf);

                rknn_L2_vs_bruteforce_test<double>(100, 2, knn, 9.0 * r * r);
                rknn_L2_vs_bruteforce_test<double>(100, 3, knn, 9.0 * r * r);
                rknn_L2_vs_bruteforce_test<double>(100, 7, knn, 9.0 * r * r);
            }
        }
    }
}

TEST(kdtree, SO3_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        SO3_vs_bruteforce_test<float>(5);

        SO3_vs_bruteforce_test<float>(100);
        SO3_vs_bruteforce_test<float>(100);
        SO3_vs_bruteforce_test<float>(100);

        SO3_vs_bruteforce_test<double>(100);
        SO3_vs_bruteforce_test<double>(100);
        SO3_vs_bruteforce_test<double>(100);
    }
}

TEST(kdtree, SO2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        SO2_vs_bruteforce_test<float>(100);
        SO2_vs_bruteforce_test<float>(100);
        SO2_vs_bruteforce_test<float>(100);

        SO2_vs_bruteforce_test<double>(100);
        SO2_vs_bruteforce_test<double>(100);
        SO2_vs_bruteforce_test<double>(100);
    }
}

TEST(kdtree, L2_concurrent_build_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        L2_concurrent_build_vs_bruteforce_test<float>(100, 2);
        L2_concurrent_build_vs_bruteforce_test<float>(100, 3);
        L2_concurrent_build_vs_bruteforce_test<float>(100, 7);

        L2_concurrent_build_vs_bruteforce_test<double>(100, 2);
        L2_concurrent_build_vs_bruteforce_test<double>(100, 3);
        L2_concurrent_build_vs_bruteforce_test<double>(100, 7);
    }
}

TEST(kdtree, L2_concurrent_build_vs_L2)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        L2_concurrent_build_vs_L2_test<float>(100, 2);
        L2_concurrent_build_vs_L2_test<float>(100, 3);
        L2_concurrent_build_vs_L2_test<float>(100, 7);

        L2_concurrent_build_vs_L2_test<double>(100, 2);
        L2_concurrent_build_vs_L2_test<double>(100, 3);
        L2_concurrent_build_vs_L2_test<double>(100, 7);
    }
}

TEST(kdtree, same_points)
{
    using num_t         = double;
    using point_cloud_t = PointCloud<num_t>;
    using kdtree_t      = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<num_t, point_cloud_t>, point_cloud_t, 3 /* dim */>;

    point_cloud_t cloud;
    cloud.pts.resize(16);
    for (size_t i = 0; i < 16; ++i)
    {
        cloud.pts[i].x = -1.;
        cloud.pts[i].y = 0.;
        cloud.pts[i].z = 1.;
    }

    kdtree_t idx(3 /*dim*/, cloud);
}

// ---------------------------------------------------------------------------
// Unsigned / integral ElementType.
//
// Building an index over an unsigned ElementType used to abort with a
// heap-buffer-overflow inside planeSplit(): the spread sentinel in
// middleSplit_() wrapped around to the largest representable value, no cut
// dimension was ever selected, and the resulting degenerate cut sent every
// point to the same side of the partition.
// ---------------------------------------------------------------------------

TEST(kdtree, unsigned_vs_bruteforce)
{
    for (uint64_t seed = 0; seed < 20; seed++)
    {
        integral_kd_vs_bruteforce_test<uint8_t, int32_t, L2_Simple_Adaptor, false>(
            200, 5, 255, seed);
        integral_kd_vs_bruteforce_test<uint8_t, int32_t, L2_Adaptor, false>(200, 5, 255, seed);
        integral_kd_vs_bruteforce_test<uint8_t, int32_t, L1_Adaptor, true>(200, 5, 255, seed);

        // A floating-point DistanceType is equally valid:
        integral_kd_vs_bruteforce_test<uint8_t, float, L2_Simple_Adaptor, false>(200, 5, 255, seed);

        integral_kd_vs_bruteforce_test<uint16_t, int64_t, L2_Simple_Adaptor, false>(
            300, 5, 65535, seed);

        // Full uint32_t range: exercises the midpoint computation in
        // middleSplit_(), which overflows if done in ElementType.
        integral_kd_vs_bruteforce_test<uint32_t, double, L2_Simple_Adaptor, false>(
            300, 5, 4294967295u, seed);
    }
}

TEST(kdtree, unsigned_radius_search)
{
    using cloud_t   = PointCloud<uint8_t>;
    using adaptor_t = L2_Simple_Adaptor<uint8_t, cloud_t, int32_t>;
    using kdtree_t  = KDTreeSingleIndexAdaptor<adaptor_t, cloud_t, 3 /* dim */>;

    cloud_t cloud;
    generateRandomIntegralPointCloud<uint8_t>(cloud, 500, 255, 42);

    kdtree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10));

    const uint8_t query_pt[3] = {100, 120, 140};
    const int32_t radius      = 10000;

    std::vector<nanoflann::ResultItem<typename kdtree_t::IndexType, int32_t>> matches;
    const size_t nFound = index.radiusSearch(&query_pt[0], radius, matches);

    // RadiusResultSet::addPoint keeps a point only if dist < radius (strict):
    size_t bf_count = 0;
    for (const auto& p : cloud.pts)
    {
        const int32_t dx = int32_t(query_pt[0]) - int32_t(p.x);
        const int32_t dy = int32_t(query_pt[1]) - int32_t(p.y);
        const int32_t dz = int32_t(query_pt[2]) - int32_t(p.z);
        if (dx * dx + dy * dy + dz * dz < radius) bf_count++;
    }

    EXPECT_GT(bf_count, 0u);
    EXPECT_EQ(nFound, bf_count);
    EXPECT_EQ(matches.size(), bf_count);
}

TEST(kdtree, unsigned_same_points)
{
    // Degenerate case: every value equals the cut value, so the partition
    // takes only the "equal" branch and must still terminate.
    using cloud_t   = PointCloud<uint8_t>;
    using adaptor_t = L2_Simple_Adaptor<uint8_t, cloud_t, int32_t>;
    using kdtree_t  = KDTreeSingleIndexAdaptor<adaptor_t, cloud_t, 3 /* dim */>;

    cloud_t cloud;
    cloud.pts.resize(16);
    for (auto& p : cloud.pts)
    {
        p.x = 200;
        p.y = 0;
        p.z = 255;
    }

    kdtree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(4));

    const uint8_t query_pt[3] = {200, 0, 255};
    size_t        ret_index   = 0;
    int32_t       out_dist    = -1;

    nanoflann::KNNResultSet<int32_t> resultSet(1);
    resultSet.init(&ret_index, &out_dist);
    ASSERT_TRUE(index.findNeighbors(resultSet, &query_pt[0]));
    EXPECT_EQ(out_dist, 0);
}

TEST(kdtree, unsigned_incremental_index)
{
    // The incremental index selects its cut axis with the same "widest
    // spread" logic, which has to stay unsigned-safe too.
    using cloud_t   = PointCloud<uint8_t>;
    using adaptor_t = L2_Simple_Adaptor<uint8_t, cloud_t, int32_t>;
    using kdtree_t  = nanoflann::KDTreeSingleIndexIncrementalAdaptor<adaptor_t, cloud_t, 3>;

    cloud_t cloud;
    generateRandomIntegralPointCloud<uint8_t>(cloud, 300, 255, 7);

    kdtree_t index(3 /*dim*/, cloud);
    index.addPoints(0, static_cast<kdtree_t::IndexType>(cloud.pts.size() - 1));

    const uint8_t query_pt[3] = {30, 60, 90};

    size_t  ret_index = 0;
    int32_t out_dist  = -1;

    nanoflann::KNNResultSet<int32_t> resultSet(1);
    resultSet.init(&ret_index, &out_dist);
    ASSERT_TRUE(index.findNeighbors(resultSet, &query_pt[0]));

    int32_t bf_best = std::numeric_limits<int32_t>::max();
    for (const auto& p : cloud.pts)
    {
        const int32_t dx = int32_t(query_pt[0]) - int32_t(p.x);
        const int32_t dy = int32_t(query_pt[1]) - int32_t(p.y);
        const int32_t dz = int32_t(query_pt[2]) - int32_t(p.z);
        bf_best          = std::min(bf_best, dx * dx + dy * dy + dz * dz);
    }

    EXPECT_EQ(out_dist, bf_best);
}
