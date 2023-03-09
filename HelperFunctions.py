import numpy as np

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
    PHI_ADJUSTMENT = np.array([0., 0., (np.pi / 2)]) - LEFT_SHOULDER_COORD
    
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