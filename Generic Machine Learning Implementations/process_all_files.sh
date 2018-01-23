#!/bin/bash




echo "Patient, AUC, AUC-Lower, AUC-Higher"
for i in *;
	do python *.py "$i";
done
