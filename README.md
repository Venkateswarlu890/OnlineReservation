
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        """
        Initialize Mobius strip parameters.
        R: radius from center to strip
        w: width of the strip
        n: resolution (number of points along u and v)
        """
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.mesh = self.compute_mesh()

    def parametric(self, u, v):
        """Parametric equations for the Mobius strip."""
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_mesh(self):
        """Compute the mesh grid of (x, y, z) points on the surface."""
        uu, vv = np.meshgrid(self.u, self.v)
        x, y, z = self.parametric(uu, vv)
        return x, y, z

    def area_element(self, u, v):
        """
        Compute the magnitude of the cross product of partial derivatives,
        which gives the local surface area element.
        """
        # Partial derivatives with respect to u
        dx_du = - (self.R + v * np.cos(u / 2)) * np.sin(u) - 0.5 * v * np.sin(u / 2) * np.cos(u)
        dy_du = (self.R + v * np.cos(u / 2)) * np.cos(u) - 0.5 * v * np.sin(u / 2) * np.sin(u)
        dz_du = 0.5 * v * np.cos(u / 2)

        # Partial derivatives with respect to v
        dx_dv = np.cos(u / 2) * np.cos(u)
        dy_dv = np.cos(u / 2) * np.sin(u)
        dz_dv = np.sin(u / 2)

        # Cross product
        cx = dy_du * dz_dv - dz_du * dy_dv
        cy = dz_du * dx_dv - dx_du * dz_dv
        cz = dx_du * dy_dv - dy_du * dx_dv

        return np.sqrt(cx**2 + cy**2 + cz**2)

    def compute_surface_area(self):
        """Numerically integrate the area element over the parameter domain."""
        area, _ = dblquad(
            self.area_element,
            -self.w/2, self.w/2,  # v limits
            lambda v: 0, lambda v: 2 * np.pi  # u limits
        )
        return area

    def compute_edge_length(self):
        """Approximate the length of the boundary (edge) numerically."""
        # Edge at v = w/2 (the Möbius strip has only one edge, but both v=±w/2 traces are used for completeness)
        u = np.linspace(0, 2 * np.pi, self.n)
        v1 = np.full_like(u, self.w/2)
        v2 = np.full_like(u, -self.w/2)
        x1, y1, z1 = self.parametric(u, v1)
        x2, y2, z2 = self.parametric(u, v2)
        # Compute distances between consecutive points and sum
        edge1 = np.sum(np.sqrt(np.diff(x1)**2 + np.diff(y1)**2 + np.diff(z1)**2))
        edge2 = np.sum(np.sqrt(np.diff(x2)**2 + np.diff(y2)**2 + np.diff(z2)**2))
        return edge1 + edge2

    def plot(self):
        """Visualize the Mobius strip."""
        x, y, z = self.mesh
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='teal', edgecolor='k', linewidth=0.1, alpha=0.8)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        ax.set_title('Mobius Strip', fontsize=16)
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=200)
    area = mobius.compute_surface_area()
    edge_length = mobius.compute_edge_length()
    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {edge_length:.4f}")
    mobius.plot()
