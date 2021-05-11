import numpy as np
from discretize.utils.matrix_utils import mkvc, ndgrid, sub2ind
from discretize.utils.code_utils import deprecate_function
import warnings


def volume_tetrahedron(xyz, A, B, C, D):
    """Returns the tetrahedron volumes for a specified set of verticies.

    Let *xyz* be an (n, 3) array denoting a set of vertex locations.
    Any 4 vertex locations *a, b, c* and *d* can be used to define a tetrahedron.
    For the set of tetrahedra whose verticies are indexed in vectors
    *A, B, C* and *D*, this function returns the corresponding volumes.
    See algorithm: https://en.wikipedia.org/wiki/Tetrahedron#Volume

    .. math::
       vol = {1 \\over 6} \\big | ( \\mathbf{a - d} ) \\cdot
       ( ( \\mathbf{b - d} ) \\times ( \\mathbf{c - d} ) ) \\big |

    Parameters
    ----------
    xyz : numpy.ndarray
        (n, 3) array containing the x,y,z locations for all verticies
    A : numpy.ndarray
        Vector containing the indicies for the **a** vertex locations
    B : numpy.ndarray
        Vector containing the indicies for the **b** vertex locations
    C : numpy.ndarray
        Vector containing the indicies for the **c** vertex locations
    D : numpy.ndarray
        Vector containing the indicies for the **d** vertex locations

    Returns
    -------
    numpy.ndarray
        Volumes of the tetrahedra whose vertices are indexes in
        *A, B, C* and *D*.


    Examples
    --------

    Here we define a small 3D tensor mesh. 4 nodes are chosen to
    be the verticies of a tetrahedron. We compute the volume of this
    tetrahedron. Note that xyz locations for the verticies can be
    scattered and do not require regular spacing.

    >>> from discretize.utils import volume_tetrahedron
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> 
    >>> mpl.rcParams.update({"font.size": 14})
    >>> 
    >>> # Corners of a uniform cube
    >>> h = [1, 1]
    >>> mesh = TensorMesh([h, h, h])
    >>> xyz = mesh.nodes
    >>> 
    >>> # Indicies
    >>> A = np.array([0])
    >>> B = np.array([6])
    >>> C = np.array([8])
    >>> D = np.array([24])
    >>> 
    >>> # Compute volume for all tetrahedra and extract first one
    >>> vol = volume_tetrahedron(xyz, A, B, C, D)
    >>> vol = vol[0]
    >>> 
    >>> # Plot
    >>> fig = plt.figure(figsize=(7, 7))
    >>> ax = fig.gca(projection='3d')
    >>> 
    >>> mesh.plot_grid(ax=ax)
    >>> 
    >>> k = [0, 6, 8, 0, 24, 6, 24, 8]
    >>> xyz_tetra = xyz[k, :]
    >>> ax.plot(xyz_tetra[:, 0], xyz_tetra[:, 1], xyz_tetra[:, 2], 'r')
    >>> 
    >>> ax.text(-0.25, 0., 3., 'Volume of the tetrahedron: {:g} $m^3$'.format(vol))
    >>> 
    >>> plt.show()

    """

    AD = xyz[A, :] - xyz[D, :]
    BD = xyz[B, :] - xyz[D, :]
    CD = xyz[C, :] - xyz[D, :]

    V = (
        (BD[:, 0] * CD[:, 1] - BD[:, 1] * CD[:, 0]) * AD[:, 2]
        - (BD[:, 0] * CD[:, 2] - BD[:, 2] * CD[:, 0]) * AD[:, 1]
        + (BD[:, 1] * CD[:, 2] - BD[:, 2] * CD[:, 1]) * AD[:, 0]
    )
    return np.abs(V / 6)


def index_cube(nodes, grid_shape, n=None):
    """Returns the index of nodes on a curvilinear mesh.



    TWO DIMENSIONS::

      node(i,j)          node(i,j+1)
           A -------------- B
           |                |
           |    cell(i,j)   |
           |        I       |
           |                |
          D -------------- C
      node(i+1,j)        node(i+1,j+1)


    THREE DIMENSIONS::

            node(i,j,k+1)       node(i,j+1,k+1)
                E --------------- F
               /|               / |
              / |              /  |
             /  |             /   |
      node(i,j,k)         node(i,j+1,k)
           A -------------- B     |
           |    H ----------|---- G
           |   /cell(i,j)   |   /
           |  /     I       |  /
           | /              | /
           D -------------- C
      node(i+1,j,k)      node(i+1,j+1,k)

    
    Parameters
    ----------
    nodes : str
        String specifying which nodes to return; e.g. 'ABCD'
    grid_shape : list
        Number of nodes along the i,j,k directions; e.g. [ni,nj,nk]
    nc : list
        Number of cells along the i,j,k directions; e.g. [nci,ncj,nck]


    Returns
    -------
    index : tuple of numpy.array
        Each entry of the tuple is a numpy array containing the indices of
        the nodes specified by the *nodes* paramter in the order asked;
        e.g. if *nodes* = 'ABCD', the tuple returned is ordered (A,B,C,D).

    """

    if not isinstance(nodes, str):
        raise TypeError("Nodes must be a str variable: e.g. 'ABCD'")
    nodes = nodes.upper()
    try:
        dim = len(grid_shape)
        if n is None:
            n = tuple(x - 1 for x in grid_shape)
    except TypeError:
        return TypeError("grid_shape must be iterable")
    # Make sure that we choose from the possible nodes.
    possibleNodes = "ABCD" if dim == 2 else "ABCDEFGH"
    for node in nodes:
        if node not in possibleNodes:
            raise ValueError("Nodes must be chosen from: '{0!s}'".format(possibleNodes))

    if dim == 2:
        ij = ndgrid(np.arange(n[0]), np.arange(n[1]))
        i, j = ij[:, 0], ij[:, 1]
    elif dim == 3:
        ijk = ndgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]))
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    else:
        raise Exception("Only 2 and 3 dimensions supported.")

    nodeMap = {
        "A": [0, 0, 0],
        "B": [0, 1, 0],
        "C": [1, 1, 0],
        "D": [1, 0, 0],
        "E": [0, 0, 1],
        "F": [0, 1, 1],
        "G": [1, 1, 1],
        "H": [1, 0, 1],
    }
    out = ()
    for node in nodes:
        shift = nodeMap[node]
        if dim == 2:
            out += (sub2ind(grid_shape, np.c_[i + shift[0], j + shift[1]]).flatten(),)
        elif dim == 3:
            out += (
                sub2ind(
                    grid_shape, np.c_[i + shift[0], j + shift[1], k + shift[2]]
                ).flatten(),
            )

    return out


def face_info(xyz, A, B, C, D, average=True, normalize_normals=True, **kwargs):
    """Returns normal surface vector and area for a given set of faces.

    Let *xyz* be an (n, 3) array denoting a set of vertex locations.
    Now let vertex locations *a, b, c* and *d* define a quadrilateral
    (regular or irregular) in 2D or 3D space. For this quadrilateral,
    we organize the vertices as follows:

    CELL VERTICES::
      
            a -------Vab------- b
           /                   /
          /                   /
        Vda       (X)       Vbc
        /                   /
       /                   /
      d -------Vcd------- c

    where the normal vector *(X)* is pointing into the page. For a set
    of quadrilaterals whose vertices are indexed in arrays *A, B, C* and *D* ,
    this function returns the normal surface vector(s) and the area
    for each quadrilateral.

    At each vertex, there are 4 cross-products that can be used to compute the
    vector normal the surface defined by the quadrilateral. In 3D space however,
    the vertices indexed may not define a quadrilateral exactly and thus the normal vectors
    computed at each vertex might not be identical. In this case, you may choose output
    the normal vector at *a, b, c* and *d* or compute
    the average normal surface vector as follows:

    .. math::
        \\bf{n} = \\frac{1}{4} \\big (
        \\bf{v_{ab} \\times v_{da}} +
        \\bf{v_{bc} \\times v_{ab}} +
        \\bf{v_{cd} \\times v_{bc}} +
        \\bf{v_{da} \\times v_{cd}} \\big )


    For computing the surface area, we assume the vertices define a quadrilateral.

    Paramters
    ---------
    xyz :
        (n, 3) array containing the x,y,z locations for all verticies
    A : numpy.ndarray
        Vector containing the indicies for the **a** vertex locations
    B : numpy.ndarray
        Vector containing the indicies for the **b** vertex locations
    C : numpy.ndarray
        Vector containing the indicies for the **c** vertex locations
    D : numpy.ndarray
        Vector containing the indicies for the **d** vertex locations
    average : bool
        If *True* (default), the function returns the average surface
        normal vector for each surface. If *False* , the function will
        return the normal vectors computed at the *A, B, C* and *D*
        vertices in a cell array {nA,nB,nC,nD}.
    normalize_normal : bool
        If *True* (default), the function will normalize the surface normal
        vectors. This is applied regardless of whether the *average* parameter
        is set to *True* or *False*. If *False*, the vectors are not normalized.

    Returns
    -------
    N : ndarray or cell array of ndarray
        Normal vector(s) for each surface. If *average* = *True*, the function
        returns an ndarray with the average surface normal vectos. If *average* = *False* ,
        the function returns a cell array {nA,nB,nC,nD} containing the normal vectors
        computed using each vertex of the surface.
    area : numpy.ndarray
        The surface areas.

    Examples
    --------

    Here we define a set of vertices for a tensor mesh. We then
    index 4 vertices for an irregular quadrilateral. The
    function *face_info* is used to compute the normal vector
    and the surface area.

    >>> from discretize.utils import face_info
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> 
    >>> mpl.rcParams.update({"font.size": 14})
    >>> 
    >>> # Corners of a uniform cube
    >>> h = [1, 1]
    >>> mesh = TensorMesh([h, h, h])
    >>> xyz = mesh.nodes
    >>> 
    >>> # Indicies
    >>> A = np.array([0])
    >>> B = np.array([4])
    >>> C = np.array([26])
    >>> D = np.array([18])
    >>> 
    >>> # Compute average surface normal vector (normalized)
    >>> nvec, area = face_info(xyz, A, B, C, D)
    >>> area = area[0]
    >>> 
    >>> # Plot
    >>> fig = plt.figure(figsize=(7, 7))
    >>> ax = fig.gca(projection='3d')
    >>> 
    >>> mesh.plot_grid(ax=ax)
    >>> 
    >>> k = [0, 4, 26, 18, 0]
    >>> xyz_quad = xyz[k, :]
    >>> ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'r')
    >>> 
    >>> ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area))
    >>> ax.text(-0.25, 0., 2.8, 'Normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
    >>>     nvec[0, 0], nvec[0, 1], nvec[0, 2])
    >>> )
    >>> 
    >>> plt.show()

    In our second example, the vertices do are unable to define a flat
    surface in 3D space. However, we will demonstrate the *face_info*
    returns the average normal vector and an approximate surface area.

    >>> # Corners of a uniform cube
    >>> h = [1, 1]
    >>> mesh = TensorMesh([h, h, h])
    >>> xyz = mesh.nodes
    >>> 
    >>> # Indicies
    >>> A = np.array([0])
    >>> B = np.array([5])
    >>> C = np.array([26])
    >>> D = np.array([18])
    >>> 
    >>> # Compute average surface normal vector
    >>> nvec, area = face_info(xyz, A, B, C, D)
    >>> area = area[0]
    >>> 
    >>> # Plot
    >>> fig = plt.figure(figsize=(7, 7))
    >>> ax = fig.gca(projection='3d')
    >>> 
    >>> mesh.plot_grid(ax=ax)
    >>> 
    >>> k = [0, 5, 26, 18, 0]
    >>> xyz_quad = xyz[k, :]
    >>> ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'g')
    >>> 
    >>> ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area))
    >>> ax.text(-0.25, 0., 2.8, 'Average normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
    >>>     nvec[0, 0], nvec[0, 1], nvec[0, 2])
    >>> )
    >>> 
    >>> plt.show()

    """
    if "normalizeNormals" in kwargs:
        warnings.warn(
            "The normalizeNormals keyword argument has been deprecated, please use normalize_normals. "
            "This will be removed in discretize 1.0.0",
            DeprecationWarning,
        )
        normalize_normals = kwargs["normalizeNormals"]
    if not isinstance(average, bool):
        raise TypeError("average must be a boolean")
    if not isinstance(normalize_normals, bool):
        raise TypeError("normalize_normals must be a boolean")
    

    AB = xyz[B, :] - xyz[A, :]
    BC = xyz[C, :] - xyz[B, :]
    CD = xyz[D, :] - xyz[C, :]
    DA = xyz[A, :] - xyz[D, :]

    def cross(X, Y):
        return np.c_[
            X[:, 1] * Y[:, 2] - X[:, 2] * Y[:, 1],
            X[:, 2] * Y[:, 0] - X[:, 0] * Y[:, 2],
            X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0],
        ]

    nA = cross(AB, DA)
    nB = cross(BC, AB)
    nC = cross(CD, BC)
    nD = cross(DA, CD)

    length = lambda x: np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    normalize = lambda x: x / np.kron(np.ones((1, x.shape[1])), mkvc(length(x), 2))
    if average:
        # average the normals at each vertex.
        N = (nA + nB + nC + nD) / 4  # this is intrinsically weighted by area
        # normalize
        N = normalize(N)
    else:
        if normalize_normals:
            N = [normalize(nA), normalize(nB), normalize(nC), normalize(nD)]
        else:
            N = [nA, nB, nC, nD]

    # Area calculation
    #
    # Approximate by 4 different triangles, and divide by 2.
    # Each triangle is one half of the length of the cross product
    #
    # So also could be viewed as the average parallelogram.
    #
    # TODO: This does not compute correctly for concave quadrilaterals
    area = (length(nA) + length(nB) + length(nC) + length(nD)) / 4

    return N, area


volTetra = deprecate_function(volume_tetrahedron, "volTetra", removal_version="1.0.0")
indexCube = deprecate_function(index_cube, "indexCube", removal_version="1.0.0")
faceInfo = deprecate_function(face_info, "faceInfo", removal_version="1.0.0")
