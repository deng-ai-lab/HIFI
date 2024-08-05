import torch
import torch.nn as nn

class Reg(nn.Module):
    """
    Regularization module for graph-based smoothness regularization.

    Args:
        net (nn.Module): The neural network model.
        group_size (int, optional): The number of inner parameters to group together. Defaults to 4.
    """

    def __init__(self, net: nn.Module, group_size: int = 4):
        super(Reg, self).__init__()
        self.group_size = group_size

        self.inner_params = [param for name, param in net.named_parameters() if 'attri' in name]
        self.outer_params = [param for name, param in net.named_parameters() if 'weight' in name and 'conv' in name]

        # Construct the nodes and adjacency matrix
        self.nodes = [self.inner_params[i:i+group_size] for i in range(0, len(self.inner_params), group_size)]
        self.adjacency = self.outer_params[1:]

        self.graph_num = len(self.adjacency)

    def forward(self):
        """
        Forward pass of the regularization module.

        Returns:
            float: The smoothness regularization loss.
        """
        smoothness = 0
        for i in range(self.graph_num):
            node1 = torch.stack([node.view(-1) for node in self.nodes[i]]).T
            node2 = torch.stack([node.view(-1) for node in self.nodes[i+1]]).T
            adj_mat = torch.einsum('ijkl->ij', [self.adjacency[i]])
            smoothness += self.laplacian(node1, node2, adj_mat)
        return smoothness

    def laplacian(self, node1: torch.Tensor, node2: torch.Tensor, adj_mat: torch.Tensor):
        """
        Compute the smoothness regularization loss.

        Args:
            node1 (torch.Tensor): The first set of nodes.
            node2 (torch.Tensor): The second set of nodes.
            adj_mat (torch.Tensor): The adjacency matrix.

        Returns:
            float: The smoothness regularization loss.
        """
        # Node concatenation
        nodes_combined = torch.cat([node1, node2], dim=0)

        # Construct new adjacency matrix
        num1 = node1.size(0)
        num2 = node2.size(0)
        num_nodes = num1 + num2
        adj_combined = torch.zeros((num_nodes, num_nodes)).cuda()
        adj_combined[:num1, num1:] = adj_mat.T
        adj_combined[num1:, :num1] = adj_mat

        # Add self-connections
        adj_combined += torch.eye(num_nodes).cuda()

        # Compute the inverse square root of the degree matrix
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj_combined, dim=1)))
        d_inv_sqrt = torch.where(torch.isnan(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)

        # Compute the symmetrically normalized Laplacian matrix L_sym
        laplacian_matrix_sym = torch.eye(num_nodes).cuda() - torch.matmul(torch.matmul(d_inv_sqrt, adj_combined), d_inv_sqrt)

        # Compute smoothness
        smoothness = torch.trace(torch.matmul(torch.matmul(nodes_combined.T, laplacian_matrix_sym), nodes_combined)) / (num_nodes * num_nodes)

        return smoothness


if __name__ == '__main__':
    from models import spiking_resnet
    from modules import neuron, surrogate as surrogate_self

    net = spiking_resnet.__dict__['spiking_resnet18'](neuron=neuron.HIFINeuron, num_classes=10, neuron_dropout=0.25,
                                                  tau=1.1, v_threshold=1.1, surrogate_function=surrogate_self.Rectangle(), c_in=3, fc_hw=1)
    
    reg = Reg(net)
    loss = reg()

    print(loss)