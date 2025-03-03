import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def plot_2d_locations(locations, title="2D Locations Plot"):
    # Extract x and y coordinates from the locations array
    y = locations[:, 0]
    x = locations[:, 1]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c="black", marker="o", s=1, alpha=0.7)

    # Add labels and title
    plt.xlabel("Longitude (Y)")
    plt.ylabel("Latitude (X)")
    plt.title(title)

    # Display the plot
    plt.show()


def plot_interactive(embedding, title, n_components=3):
    if n_components == 3:
        x = embedding[:, 0]
        y = embedding[:, 1]
        z = embedding[:, 2]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=1.5, color=z, colorscale="matter", opacity=0.8),
                )
            ]
        )

        fig.update_layout(
            title=title, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
        )
        fig.show()
    elif n_components == 2:
        x = embedding[:, 0]
        y = embedding[:, 1]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=5, color=y, colorscale="matter", opacity=0.8),
                )
            ]
        )

        fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
        fig.show()


def plot_interactive_with_links(embedding, title, k=5, n_components=3):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embedding)
    _, neighbors = nn.kneighbors(embedding)

    if n_components == 3:
        x = embedding[:, 0]
        y = embedding[:, 1]
        z = embedding[:, 2]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=3, color=z, colorscale="matter", opacity=0.8),
                name="Points",
            )
        )

        for i in range(len(embedding)):
            for j in neighbors[i]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[x[i], x[j]],
                        y=[y[i], y[j]],
                        z=[z[i], z[j]],
                        mode="lines",
                        line=dict(color="lightblue", width=1),
                        name="Neighbor Link",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title=title, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
        )
        fig.show()

    elif n_components == 2:
        x = embedding[:, 0]
        y = embedding[:, 1]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=5, color=y, colorscale="matter", opacity=0.8),
                name="Points",
            )
        )

        for i in range(len(embedding)):
            for j in neighbors[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i], x[j]],
                        y=[y[i], y[j]],
                        mode="lines",
                        line=dict(color="lightblue", width=1),
                        name="Neighbor Link",
                        showlegend=False,
                    )
                )

        fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
        fig.show()


def plot_subplots(embeddings, titles, plot_shape=(3, 3), n_components=3):
    fig = plt.figure(figsize=(10, 10))

    assert len(embeddings) == len(
        titles
    ), "Length of embeddings and titles must be the same"
    assert (
        len(embeddings) == plot_shape[0] * plot_shape[1]
    ), "Number of embeddings must match the plot shape"

    for i, (embedding, title) in enumerate(zip(embeddings, titles)):
        ax = fig.add_subplot(
            plot_shape[0],
            plot_shape[1],
            i + 1,
            projection="3d" if n_components == 3 else None,
        )

        if n_components == 3:
            x = embedding[:, 0]
            y = embedding[:, 1]
            z = embedding[:, 2]
            ax.scatter(x, y, z, c=z, cmap="cool", s=1, alpha=0.8)  # type: ignore
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")  # type: ignore
        elif n_components == 2:
            x = embedding[:, 0]
            y = embedding[:, 1]
            ax.scatter(x, y, c=y, cmap="cool", s=5, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()
