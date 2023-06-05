#!/bin/bash

for agent in h a
do
	for m_file in model_udiff.py model_adiff.py model_fdiff.py
	do
		# Table 2
		python3 $m_file $agent fb 1
		# Table 6
		python3 $m_file $agent nofb 1
		# Tables 7 and 8
		python3 $m_file $agent both 0
	done
done

