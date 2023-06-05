#!/bin/bash

for agent in h a
do
	for m_file in model_1d.py model_md.py
	do
		# Table 1
		python3 $m_file other $agent fb 1
		# Tables 4 and 5
		python3 $m_file other $agent both 0
	done
done


