#!/bin/bash

#--- Switch to the correct path (where the script is in)
backpath="$(pwd)"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#--- Download the compressed file
#    Try the updated Princeton URL first, fall back to alternative mirror
wget -nc http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip || \
  wget -nc http://modelnet.cs.princeton.edu/ModelNet10.zip

#--- Extract it
unzip -n ModelNet10.zip

#--- Go back to the initial directory
cd "$backpath"
