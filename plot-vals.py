import os
import matplotlib.pyplot as plt
from optparse import OptionParser

# Use the optparse module for passing the filename on the command line
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="data filename")
(options, args) = parser.parse_args()

# Get the base name of the given file
figname = os.path.splitext(options.filename)[0]

# Open the file and parse the input into an array of floats
# This will skip the first line since that is assumbed to be the matrix dimension
with open(options.filename) as f:
    data = f.readlines()

parsed_input = []
for i in data[1:]:
    for j in i.rstrip('\n').split(' '):
        if j:
            parsed_input.append(float(j))

# Create a figure handle
fig = plt.figure()

# Plot the values with generic x and y labels
plt.plot(range(0, len(parsed_input)), parsed_input)
plt.ylabel('Y')
plt.xlabel('X')
# Make the title that of the given filename
plt.title(options.filename)
plt.axis([0, len(parsed_input), min(parsed_input), max(parsed_input)])
plt.grid(True)

# Save the figure as a png file
fig.savefig(figname + ".png")

# Show the plot
plt.show()
