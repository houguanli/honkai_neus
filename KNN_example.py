import frnn
import torch

source_point_num = 1000
query_point_num = 10
dim = 3

# genetate random point cloud
def gen_pc(n):
    return 200 * torch.rand(n, 3).cuda()

source_point = gen_pc(source_point_num)
query_point = gen_pc(query_point_num)

# N, P1, dim
# N: Num of Point cloud
# P1: How many points are contained in point clouds
# dim: spatial dimension
source_point = source_point.reshape(1, source_point_num, dim)
query_point = query_point.reshape(1, query_point_num, dim)

max_length = 100
neighbor_num = 2

# dists: distance
# idxs: index(the near(est) point(s) index in source point cloud for query points)
# nn: index(the near(est) point(s) index in query point cloud for source points)
# grid: constructed once for spartial discretization and point querying.
dists, idxs, nn, grid = frnn.frnn_grid_points(
        query_point, source_point, None, None, neighbor_num, max_length, grid=None, return_nn=False, return_sorted=True
)

# if the source_point has not been changed, reuse "grid".

# dists, idxs, nn, grid = frnn.frnn_grid_points(
#         query_point, source_point, None, None, 1, 100, grid=grid, return_nn=False, return_sorted=True
# )

# idx sae
print(idxs.shape)

# print(source_point[0,865])
import pdb; pdb.set_trace()
for i in range(0, query_point_num):
    for j in range(0, neighbor_num):
        print(f'The {j+1}th nearest point in source point cloud to the {i+1}th point in query point could is:', source_point[0, idxs[0, i, j]])
        print("corresponding dist: ", dists[0, i, j])
