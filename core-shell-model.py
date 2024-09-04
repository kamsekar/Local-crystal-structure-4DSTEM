# Turns an ordered Pt-Cu alloy nanoparticle model to a core-shell model with a
# disordered alloy core, an ordered alloy shell, and a Pt-rich surface.
# Saves the model as an .xyz file.
# ana.rebeka.kamsek@ki.si, 2024

import numpy as np

# import a spherical nanoparticle with an ordered PtCu3 alloy structure
filename = r"name-of-model.xyz"
f = open(filename, "r")

# read contents of the file
lines = f.readlines()

# create a list for elements, which will be modified later
element = []
# create tuples for spatial coordinates, which will remain unmodified
x, y, z = (), (), ()

for i, line in enumerate(lines):
    if i > 1:
        element.append(line.split()[0])
        x = x + (float(line.split()[1]),)
        y = y + (float(line.split()[2]),)
        z = z + (float(line.split()[3]),)
f.close()

# determine the outer radius and entire volume
r_outer = (max(x) - min(x)) / 2
volume = np.power(r_outer, 3) * 4 * np.pi / 3

# determine the inner radius for the core-shell structure
# with a pre-defined volume fraction of the ordered phase
x_ordered = 0.7
x_disordered = 1 - x_ordered

volume_disordered = x_disordered * volume
r_inner = np.power(volume_disordered * 3 / (4 * np.pi), 0.33)

# determine a radius where the Pt-rich surface begins
r_Pt = 0.9 * r_outer

# all atoms within the inner radius are randomly assigned Cu or Pt
# and all atoms outside of a specified radius are Pt
for i in range(len(element)):
    if x[i] ** 2 + y[i] ** 2 + z[i] ** 2 <= r_inner ** 2:
        if np.random.uniform() > 0.25:
            element[i] = 'Cu'
        else:
            element[i] = 'Pt'
    elif x[i] ** 2 + y[i] ** 2 + z[i] ** 2 > r_Pt ** 2:
        element[i] = 'Pt'

# save the model to an .xyz file
with open(r'name-of-new-file.xyz', 'w') as new_file:
    new_file.write('{}\n'.format(len(element)))
    new_file.write('Cu3 Pt\n')
    for i in range(len(element)):
        new_file.write('{} {} {} {}\n'.format(element[i], x[i], y[i], z[i]))
new_file.close()
