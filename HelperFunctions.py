import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def cartesianToSpherical(cartesianCoordinate):
    assert cartesianCoordinate.shape[0] == 3, "Make sure your input is a numpy array of length 3."
    assert cartesianCoordinate.dtype == np.dtype('float64'), "Make sure all inputs are floats!"
    
    x = cartesianCoordinate[0]
    y = cartesianCoordinate[1]
    z = cartesianCoordinate[2]

    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rho)
    phi = np.arctan2(y, x)
    return np.array([rho, theta, phi])

def sphericalToCartesian(sphericalCoordinate):
    assert sphericalCoordinate.shape[0] == 3, "Make sure your input is a numpy array of length 3."
    assert sphericalCoordinate.dtype == np.dtype('float64'), "Make sure all inputs are floats!"
    
    rho = sphericalCoordinate[0]
    theta = sphericalCoordinate[1]
    phi = sphericalCoordinate[2]

    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return np.array([x, y, z])

# This function takes a landmark and an index, and returns a simple [x, y, z] numpy array of the location.
def getLocation(landmark, index):
    return np.array([landmark[index].x, landmark[index].y, landmark[index].z])

# This function converts a data landmark into a formatted feature vector.
def getFeatureVector(landmark):
    # We need to:
    #    • Put values relative to our chest location 
    #    • Convert our location values to spherical
    
    # There isn't a "chest" location, so let's put it to be halfway between the two shoulders.
    CHEST_LOCATION_CARTESIAN = (getLocation(landmark, 11) + getLocation(landmark, 12)) / 2.0
    
    # And we want to rotate our image along the phi-axis, so that the angle of the left shoulder is π/2 = 90°.
    # This means that the angle of the right shoulder, since our center location is halfway between, should be 3π/2 = 180°.
    LEFT_SHOULDER_COORD = cartesianToSpherical(getLocation(landmark, 11) - CHEST_LOCATION_CARTESIAN)
    PHI_ADJUSTMENT = np.array([0., 0., (np.pi / 2) - LEFT_SHOULDER_COORD[2]])
    
    # We don't care about head locations aside from the nose, so here are the indexes we care about.
    indexesToUse = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    formattedRow = np.empty(0)
    for index in indexesToUse:
        # Let's get the two angles mod 2π.
        newTriple = cartesianToSpherical(getLocation(landmark, index) - CHEST_LOCATION_CARTESIAN) + PHI_ADJUSTMENT
        newTriple[1] = newTriple[1] % (2 * np.pi)
        newTriple[2] = newTriple[2] % (2 * np.pi)
        formattedRow = np.append(formattedRow, newTriple)
    return formattedRow


# This is a temporary Coordinate and Landmark class I can use to test my code with.
class Coordinate:
    def __init__(self, x_in, y_in, z_in):
        self.x = x_in
        self.y = y_in
        self.z = z_in
        
    def value(self):
        return np.array([self.x, self.y, self.z])
        
# Given a file name, this will output the landmark you can run through getFeatureVector().
def getLandmark(messageFilename):
    with open(messageFilename) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = lines[i].replace('[','')
            lines[i] = lines[i].replace('\n','')
            lines[i] = lines[i].replace('x','')
            lines[i] = lines[i].replace('y','')
            lines[i] = lines[i].replace('z','')
            lines[i] = lines[i].replace(':','')
            lines[i] = lines[i].replace(' ','')
            lines[i] = lines[i].replace(',','')
        lines = [x for i, x in enumerate(lines) if i%4 != 3]
        lines = lines[:-1]
        for i, line in enumerate(lines):
            lines[i] = (float)(lines[i])

    locations = np.ndarray((33,), dtype = np.object)
    for i in range(33):
        locations[i] = Coordinate(lines[3*i], lines[3*i + 1], lines[3*i + 2])
    return locations

# This should plot our feature vector as a 3d array.
def plot(fv):
    fv.shape = 23, 3

    fvCartesian = np.empty((23, 3))
    for i, row in enumerate(fv):
        fvCartesian[i] = sphericalToCartesian(row)
#     return fvCartesian

    fig = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection='3d')
    
#     Let's plot little dots on the edges.
    x = [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
    y = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
    z = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
    ax.scatter3D(x, y, z, c = "black", alpha = 1)
    
    # Let's swap x and z.
    fvCartesian[:, [2, 0]] = fvCartesian[:, [0, 2]]
    
    # And plot the coordinates:
    ax.scatter3D(fvCartesian[:,0], fvCartesian[:,1], fvCartesian[:,2], c = "white", s = 2.5, alpha = 1)
    ax.scatter3D(fvCartesian[:,0], fvCartesian[:,1], fvCartesian[:,2], c = "black", s = 4.5, alpha = 0.5)
    ax.scatter3D(fvCartesian[0,0], fvCartesian[0,1], fvCartesian[0,2], c = "black", s = 800, alpha = 0.5)
    ax.scatter3D(fvCartesian[0,0], fvCartesian[0,1], fvCartesian[0,2], c = "black", s = 600, alpha = 0.75)
    
    # Now, let's draw connections.
    # Let's draw the shoulder, nose, and hip connections in white
    halfConnections = [(0, 1), (0, 2), (1, 2), (13, 14)]
    for connection in halfConnections:
        rowA = fvCartesian[connection[0]]
        rowB = fvCartesian[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#444444", lw = 2, alpha = 1)
        
    # And the left-side connections, in cyan
    leftHalfConnections = [(1, 3), (3, 5), (5, 7), (7, 9), (5, 9), (5, 11),
                           (1, 13), (13, 15), (15, 17), (17, 19), (19, 21), (17, 21)]
    for connection in leftHalfConnections:
        rowA = fvCartesian[connection[0]]
        rowB = fvCartesian[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#00CCCC", lw = 2.5, alpha = 1)
        
    # And the right-side connections, in cyan
    rightHalfConnections = [(2, 4), (4, 6), (6, 8), (8, 10), (6, 10), (6, 12),
                            (2, 14), (14, 16), (16, 18), (18, 20), (20, 22), (18, 22)]
    for connection in rightHalfConnections:
        rowA = fvCartesian[connection[0]]
        rowB = fvCartesian[connection[1]]
        mat = np.transpose([rowA, rowB])
        ax.plot3D(mat[0], mat[1], mat[2], c = "#EEA000", lw = 2.5, alpha = 1)

    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim = 180, elev = 0, roll = 0)
    ax.set_box_aspect(aspect = (1,1,1))
    plt.show()