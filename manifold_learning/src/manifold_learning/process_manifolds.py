import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Define the Implicit Neural Network as a Class
# =============================================================================
class ImplicitNet(nn.Module):
    """
    A simple multilayer perceptron (MLP) that takes a 3D point as input and outputs
    a scalar value. The goal is to learn an implicit function F(x,y,z) such that
    F(x,y,z)=0 defines the surface.
    """

    def __init__(
        self, in_features=3, hidden_features=128, num_hidden_layers=3, out_features=1
    ):
        super(ImplicitNet, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2. Define a Class to Encapsulate the SDF Model, Training, and Inference
# =============================================================================
class SDFModel:
    """
    Encapsulates the implicit function network along with training and inference methods.
    """

    def __init__(self, hidden_features=128, num_hidden_layers=3, lr=1e-3, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ImplicitNet(
            in_features=3,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            out_features=1,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(
        self, points, sdf_values, num_epochs=2000, batch_size=256, print_every=200
    ):
        """
        Train the SDF model on provided data.

        Parameters:
          points     : Tensor of shape (N, 3) with 3D coordinates.
          sdf_values : Tensor of shape (N, 1) with the corresponding SDF values.
          num_epochs : Number of training epochs.
          batch_size : Mini-batch size.
          print_every: Frequency of printing training loss.
        """
        N = points.shape[0]
        for epoch in range(num_epochs):
            # Sample a mini-batch
            indices = torch.randint(0, N, (batch_size,))
            batch_points = points[indices].to(self.device)
            batch_sdf = sdf_values[indices].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_points)
            loss = self.loss_fn(predictions, batch_sdf)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    def evaluate(self, points):
        """
        Evaluate the model on given points.

        Parameters:
          points: Tensor of shape (N, 3).

        Returns:
          Tensor of shape (N, 1) with predicted SDF values.
        """
        self.model.eval()
        with torch.no_grad():
            points = points.to(self.device)
            predictions = self.model(points)
        return predictions

    def compute_normal(self, point):
        """
        Compute the surface normal at a given point on the surface (i.e., where F(point) ~ 0).
        Uses automatic differentiation to compute the gradient and normalizes it.

        Parameters:
          point: A list or 1D tensor with 3 coordinates (x,y,z).
                 (It will be converted to a 2D tensor of shape (1,3) with requires_grad=True.)

        Returns:
          normal: A tensor with shape (3,) representing the unit normal vector at the point.
        """
        # Ensure point is a tensor of shape (1,3) and on the proper device
        if not torch.is_tensor(point):
            point = torch.tensor(point, dtype=torch.float32)
        point = point.view(1, 3).to(self.device)
        point.requires_grad_(True)

        self.model.train()  # Ensure gradients can be computed
        F_val = self.model(point)
        # Compute gradient: dF/dx, dF/dy, dF/dz
        grad_F = torch.autograd.grad(
            outputs=F_val,
            inputs=point,
            grad_outputs=torch.ones_like(F_val),
            create_graph=True,
        )[0]
        # Normalize the gradient to get the unit normal
        normal = grad_F[0] / (torch.norm(grad_F[0]) + 1e-8)
        return normal.cpu()

    def plot_sdf_slice(
        self, axis="x", fixed_coords=(0.0, 0.0), x_range=(-1.5, 1.5), num_points=300
    ):
        """
        Plot the learned SDF along a 1D slice. For example, along the x-axis with y=z fixed.

        Parameters:
          axis       : The axis along which to vary ('x', 'y', or 'z').
          fixed_coords: A tuple with the fixed values for the other two coordinates.
          x_range    : Tuple defining the range for the variable coordinate.
          num_points : Number of points in the slice.
        """
        self.model.eval()
        xs = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1)
        if axis == "x":
            coords = torch.cat(
                [
                    xs,
                    torch.full_like(xs, fixed_coords[0]),
                    torch.full_like(xs, fixed_coords[1]),
                ],
                dim=1,
            )
        elif axis == "y":
            coords = torch.cat(
                [
                    torch.full_like(xs, fixed_coords[0]),
                    xs,
                    torch.full_like(xs, fixed_coords[1]),
                ],
                dim=1,
            )
        elif axis == "z":
            coords = torch.cat(
                [
                    torch.full_like(xs, fixed_coords[0]),
                    torch.full_like(xs, fixed_coords[1]),
                    xs,
                ],
                dim=1,
            )
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        coords = coords.to(self.device)
        with torch.no_grad():
            sdf_vals = self.model(coords)
        plt.figure(figsize=(8, 5))
        plt.plot(xs.cpu().numpy(), sdf_vals.cpu().numpy(), label="Learned SDF")
        plt.xlabel(f"{axis}")
        plt.ylabel("F")
        plt.title(f"SDF slice along {axis}-axis")
        plt.legend()
        plt.show()


# =============================================================================
# 3. Utility Function: Generate Synthetic SDF Data for a Sphere
# =============================================================================
def generate_sphere_sdf_data(N=10000, cube_size=3.0):
    """
    Generates N random points in a cube centered at the origin and computes the signed
    distance to a sphere of radius 1 (centered at the origin).

    Returns:
      points    : Tensor of shape (N, 3) with coordinates.
      sdf_values: Tensor of shape (N, 1) with signed distance values.
    """
    # Uniformly sample points in the cube [-cube_size/2, cube_size/2]^3
    points = (torch.rand(N, 3) - 0.5) * cube_size
    # True SDF for a sphere of radius 1: sqrt(x^2+y^2+z^2) - 1.
    sdf_values = torch.norm(points, dim=1, keepdim=True) - 1.0
    return points, sdf_values


# =============================================================================
# 4. Main Function to Train and Evaluate the Model
# =============================================================================
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate synthetic training data for a sphere SDF.
    points, sdf_values = generate_sphere_sdf_data(N=10000, cube_size=3.0)

    # Initialize the SDF model.
    sdf_model = SDFModel(
        hidden_features=128, num_hidden_layers=3, lr=1e-3, device=device
    )

    # Train the model.
    sdf_model.train(
        points, sdf_values, num_epochs=2000, batch_size=256, print_every=200
    )

    # Evaluate and plot the SDF along the x-axis at y=z=0.
    sdf_model.plot_sdf_slice(
        axis="x", fixed_coords=(0.0, 0.0), x_range=(-1.5, 1.5), num_points=300
    )

    # Compute and print the normal at a test point (for a sphere, (1,0,0) lies on the surface).
    test_point = [1.0, 0.0, 0.0]
    normal = sdf_model.compute_normal(test_point)
    print(f"At point {test_point}, the computed unit normal is: {normal.numpy()}")


if __name__ == "__main__":
    main()
