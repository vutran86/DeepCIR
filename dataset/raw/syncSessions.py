import numpy as np 
import struct
import glob
import re
import sys
import os
import matplotlib.pyplot as plt
import pickle

META_SIZE = 78
DIAG_SIZE = 20
CIR_SIZE = 120*4

test_day = sys.argv[1]
traj_id = sys.argv[2]
stamp = sys.argv[3]

#========================= READ ANC LOC =========================
with open(f"{test_day}/{traj_id}/gt/{traj_id}_ancloc.csv") as viconfile:
	line = viconfile.readline()
	attrs = line.rstrip().split(",")
	anc0_loc = np.array([float(attrs[0]), float(attrs[1]), float(attrs[2])])
	line = viconfile.readline()
	attrs = line.rstrip().split(",")
	anc1_loc = np.array([float(attrs[0]), float(attrs[1]), float(attrs[2])])
	line = viconfile.readline()
	attrs = line.rstrip().split(",")
	anc2_loc = np.array([float(attrs[0]), float(attrs[1]), float(attrs[2])])
print(anc0_loc)
print(anc1_loc)
print(anc2_loc)
print(stamp)

#========================= READ GT =========================
tag0_gt = {}
with open(f"{test_day}/{traj_id}/gt/{traj_id}_tagloc.csv") as gtfile:
	for line in gtfile:
		attrs = line.rstrip().split(",")
		tval = int(attrs[1])
		tag0_gt[tval] = np.array([float(attrs[2]), float(attrs[3]), float(attrs[4])])
print(len(tag0_gt))

#========================= SYNC CIR =========================
def readTag(file_list):
	results = {}
	sorted_file_list = sorted(file_list)
	for filename in sorted_file_list:
		with open(filename, "rb") as logfile:
			logdata = logfile.read()
			meta = struct.unpack("QBBBBIIIIIIIII", logdata[:48]) #48-78: 30 bytes for timestamps
			if meta[1] != 0:
				print("Wrong tag address")
			diag0 = np.frombuffer(logdata[META_SIZE + 0*(DIAG_SIZE + CIR_SIZE):META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE], dtype=np.uint16)
			diag1 = np.frombuffer(logdata[META_SIZE + 1*(DIAG_SIZE + CIR_SIZE):META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE], dtype=np.uint16)
			diag2 = np.frombuffer(logdata[META_SIZE + 2*(DIAG_SIZE + CIR_SIZE):META_SIZE + 2*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE], dtype=np.uint16)
			resp0 = np.frombuffer(logdata[META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE: META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE + CIR_SIZE], dtype=np.int16)
			resp0 = resp0.astype(np.float32).reshape((resp0.shape[0]//2, 2))
			resp1 = np.frombuffer(logdata[META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE: META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE + CIR_SIZE], dtype=np.int16)
			resp1 = resp1.astype(np.float32).reshape((resp1.shape[0]//2, 2))
			resp2 = np.frombuffer(logdata[META_SIZE + 2*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE: META_SIZE + 2*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE + CIR_SIZE], dtype=np.int16)
			resp2 = resp2.astype(np.float32).reshape((resp2.shape[0]//2, 2))
			tval = meta[0]
			if tval in results:
				print("======== Duplicated TVAL ========")
			tfn = re.sub(".dat", "", re.sub(".*/", "", filename))
			results[tval] = {'meta': meta, 'diag_anc0': diag0, 'diag_anc1': diag1, 'diag_anc2': diag2, 'resp_anc0': resp0, 'resp_anc1': resp1, 'resp_anc2': resp2, 'tfn': tfn}
	return results

def readAnch(file_list):
	results = {}
	sorted_file_list = sorted(file_list)
	for filename in sorted_file_list:
		with open(filename, "rb") as logfile:
			logdata = logfile.read()
			meta = struct.unpack("QBBBBIIIIIIIII", logdata[:48])
			diag0 = np.frombuffer(logdata[META_SIZE + 0*(DIAG_SIZE + CIR_SIZE):META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE], dtype=np.uint16)
			diag1 = np.frombuffer(logdata[META_SIZE + 1*(DIAG_SIZE + CIR_SIZE):META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE], dtype=np.uint16)	
			poll = np.frombuffer(logdata[META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE: META_SIZE + 0*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE + CIR_SIZE], dtype=np.int16)
			poll = poll.astype(np.float32).reshape((poll.shape[0]//2, 2))
			final = np.frombuffer(logdata[META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE: META_SIZE + 1*(DIAG_SIZE + CIR_SIZE) + DIAG_SIZE + CIR_SIZE], dtype=np.int16)
			final = final.astype(np.float32).reshape((final.shape[0]//2, 2))
			tval = meta[0]
			if tval in results:
				print("======== Duplicated TVAL ========")
			afn = re.sub(".dat", "", re.sub(".*/", "", filename))
			results[tval] = {'meta': meta, 'diag0': diag0, 'diag1': diag1, 'poll': poll, 'final': final, 'afn': afn}
	return results

tag0_dir = f"{test_day}/{traj_id}/tag0/log-tag0-{stamp}"
tag0_files = glob.glob(tag0_dir + "/*.dat")
tag0_data = readTag(tag0_files)
prev_k = None
for k in sorted(tag0_data):
	if (prev_k is not None) and (prev_k >= k):
		print("Invalid data")
	prev_k = k
print(len(tag0_data))
anc0_dir = f"{test_day}/{traj_id}/anc0/log-anc0-{stamp}"
anc0_files = glob.glob(anc0_dir + "/*.dat")
anc0_data = readAnch(anc0_files)
prev_k = None
for k in sorted(anc0_data):
	if (prev_k is not None) and (prev_k >= k):
		print("Invalid data")
	prev_k = k
print(len(anc0_data))
anc1_dir = f"{test_day}/{traj_id}/anc1/log-anc1-{stamp}"
anc1_files = glob.glob(anc1_dir + "/*.dat")
anc1_data = readAnch(anc1_files)
prev_k = None
for k in sorted(anc1_data):
	if (prev_k is not None) and (prev_k >= k):
		print("Invalid data")
	prev_k = k
print(len(anc1_data))
anc2_dir = f"{test_day}/{traj_id}/anc2/log-anc2-{stamp}"
anc2_files = glob.glob(anc2_dir + "/*.dat")
anc2_data = readAnch(anc2_files)
prev_k = None
for k in sorted(anc2_data):
	if (prev_k is not None) and (prev_k >= k):
		print("Invalid data")
	prev_k = k
print(len(anc2_data))

tag0_tvals = sorted(list(tag0_data.keys()))
anc0_tvals = sorted(list(anc0_data.keys()))
anc1_tvals = sorted(list(anc1_data.keys()))
anc2_tvals = sorted(list(anc2_data.keys()))

sync_data = []
while (len(tag0_tvals) > 0) and (len(anc0_tvals) > 0) and (len(anc1_tvals) > 0) and (len(anc2_tvals) > 0):
	tag0_valid = tag0_data[tag0_tvals[0]]['meta'][2]
	tag0_valid_anc0 = tag0_valid & 0x01
	tag0_valid_anc1 = tag0_valid & 0x02
	tag0_valid_anc2 = tag0_valid & 0x04
	tag0_seq_anc0 = tag0_data[tag0_tvals[0]]['diag_anc0'][8]
	tag0_seq_anc1 = tag0_data[tag0_tvals[0]]['diag_anc1'][8]
	tag0_seq_anc2 = tag0_data[tag0_tvals[0]]['diag_anc2'][8]
	cur_seq_anc0 = (tag0_seq_anc0 + 1) % 256
	cur_seq_anc1 = (tag0_seq_anc1 + 1) % 256
	cur_seq_anc2 = (tag0_seq_anc2 + 1) % 256

	if tag0_valid_anc0:
		index = 0 
		anc0_seq0 = anc0_data[anc0_tvals[index]]['diag0'][8]
		anc0_seq1 = anc0_data[anc0_tvals[index]]['diag1'][8]
		assert anc0_seq0 == anc0_seq1, "anc0_seq0 is not equal to anc0_seq1"
		while (len(anc0_tvals) > 0) and (index < len(anc0_tvals) - 1) and ((anc0_tvals[index] - tag0_tvals[0] <= -2*1000000) or \
			((anc0_tvals[index] - tag0_tvals[0] < 2*1000000) and (((cur_seq_anc0 - anc0_seq0 > 0) and (cur_seq_anc0 - anc0_seq0 < 50)) or ((cur_seq_anc0 - anc0_seq0 < 0) and (cur_seq_anc0 - anc0_seq0 < -205))) ) ):
			index += 1
			anc0_seq0 = anc0_data[anc0_tvals[index]]['diag0'][8]
			anc0_seq1 = anc0_data[anc0_tvals[index]]['diag1'][8]
			assert anc0_seq0 == anc0_seq1, "anc0_seq0 is not equal to anc0_seq1"
		for i in range(index):
			anc0_tvals.pop(0)
		anc0_seq0 = anc0_data[anc0_tvals[0]]['diag0'][8] #Make sure the correct tval and seq
		anc0_seq1 = anc0_data[anc0_tvals[0]]['diag1'][8]
		assert anc0_seq0 == anc0_seq1, "anc0_seq0 is not equal to anc0_seq1"

	if tag0_valid_anc1:
		index = 0 
		anc1_seq0 = anc1_data[anc1_tvals[index]]['diag0'][8]
		anc1_seq1 = anc1_data[anc1_tvals[index]]['diag1'][8]
		assert anc1_seq0 == anc1_seq1, "anc1_seq0 is not equal to anc1_seq1"
		while (len(anc1_tvals) > 0) and (index < len(anc1_tvals) - 1) and ((anc1_tvals[index] - tag0_tvals[0] <= -2*1000000) or \
			((anc1_tvals[index] - tag0_tvals[0] < 2*1000000) and (((cur_seq_anc1 - anc1_seq0 > 0) and (cur_seq_anc1 - anc1_seq0 < 50)) or ((cur_seq_anc1 - anc1_seq0 < 0) and (cur_seq_anc1 - anc1_seq0 < -205))) ) ):
			index += 1
			anc1_seq0 = anc1_data[anc1_tvals[index]]['diag0'][8]
			anc1_seq1 = anc1_data[anc1_tvals[index]]['diag1'][8]
			assert anc1_seq0 == anc1_seq1, "anc1_seq0 is not equal to anc1_seq1"
		for i in range(index):
			anc1_tvals.pop(0)
		anc1_seq0 = anc1_data[anc1_tvals[0]]['diag0'][8] #Make sure the correct tval and seq
		anc1_seq1 = anc1_data[anc1_tvals[0]]['diag1'][8]
		assert anc1_seq0 == anc1_seq1, "anc1_seq0 is not equal to anc1_seq1"

	if tag0_valid_anc2:
		index = 0 
		anc2_seq0 = anc2_data[anc2_tvals[index]]['diag0'][8]
		anc2_seq1 = anc2_data[anc2_tvals[index]]['diag1'][8]
		assert anc2_seq0 == anc2_seq1, "anc2_seq0 is not equal to anc2_seq1"
		while (len(anc2_tvals) > 0) and (index < len(anc2_tvals) - 1) and ((anc2_tvals[index] - tag0_tvals[0] <= -2*1000000) or \
			((anc2_tvals[index] - tag0_tvals[0] < 2*1000000) and (((cur_seq_anc2 - anc2_seq0 > 0) and (cur_seq_anc2 - anc2_seq0 < 50)) or ((cur_seq_anc2 - anc2_seq0 < 0) and (cur_seq_anc2 - anc2_seq0 < -205))) ) ):
			index += 1
			anc2_seq0 = anc2_data[anc2_tvals[index]]['diag0'][8]
			anc2_seq1 = anc2_data[anc2_tvals[index]]['diag1'][8]
			assert anc2_seq0 == anc2_seq1, "anc2_seq0 is not equal to anc2_seq1"
		for i in range(index):
			anc2_tvals.pop(0)
		anc2_seq0 = anc2_data[anc2_tvals[0]]['diag0'][8] #Make sure the correct tval and seq
		anc2_seq1 = anc2_data[anc2_tvals[0]]['diag1'][8]
		assert anc2_seq0 == anc2_seq1, "anc2_seq0 is not equal to anc2_seq1"

	sample = [-1]*7
	sample_empty = True
	seq_num = -1
	if (tag0_valid_anc0 > 0) and (cur_seq_anc0 == anc0_seq0) and (tag0_tvals[0] in tag0_gt):
		seq_num = cur_seq_anc0
		sample_empty = False
	if (tag0_valid_anc1 > 0) and (cur_seq_anc1 == anc1_seq0) and (tag0_tvals[0] in tag0_gt):
		seq_num = cur_seq_anc1
		sample_empty = False
	if (tag0_valid_anc2 > 0) and (cur_seq_anc2 == anc2_seq0) and (tag0_tvals[0] in tag0_gt):
		seq_num = cur_seq_anc2
		sample_empty = False
	#======== validate ==========
	if (tag0_valid_anc0 > 0) and (cur_seq_anc0 == anc0_seq0) and (tag0_tvals[0] in tag0_gt):
		if seq_num != anc0_seq0:
			print("Invalida data")
	if (tag0_valid_anc1 > 0) and (cur_seq_anc1 == anc1_seq0) and (tag0_tvals[0] in tag0_gt):
		if seq_num != anc1_seq0:
			print("Invalid data")
	if (tag0_valid_anc2 > 0) and (cur_seq_anc2 == anc2_seq0) and (tag0_tvals[0] in tag0_gt):
		if seq_num != anc2_seq0:
			print("Invalid data")

	sample = {'gt': None, 'tval': None, 'seq': -1, 'anc0': None, 'anc1': None, 'anc2': None}
	tag = tag0_data[tag0_tvals[0]]
	fp_resp_anc0 = float((tag['diag_anc0'])[0])/64.0
	fp_resp_anc1 = float((tag['diag_anc1'])[0])/64.0
	fp_resp_anc2 = float((tag['diag_anc2'])[0])/64.0
	range_anc0 = tag['meta'][5]
	range_anc1 = tag['meta'][6]
	range_anc2 = tag['meta'][7]

	if (tag0_tvals[0] in tag0_gt):
		loc_gt = tag0_gt[tag0_tvals[0]]

	if (tag0_valid_anc0 > 0) and (seq_num == anc0_seq0) and (tag0_tvals[0] in tag0_gt):
		anc0 = anc0_data[anc0_tvals[0]]
		fp_poll_anc0 = float((anc0['diag0'])[0])/64.0
		fp_final_anc0 = float((anc0['diag1'])[0])/64.0
		gt_to_anc0 = loc_gt - anc0_loc
		gt_to_anc0 = np.sqrt(gt_to_anc0[0]**2 + gt_to_anc0[1]**2 + gt_to_anc0[2]**2)
		sample['anc0'] = {'poll': anc0['poll'], 'fp_poll': fp_poll_anc0, 'resp': tag['resp_anc0'], 'fp_resp': fp_resp_anc0, 'final': anc0['final'], 'fp_final': fp_final_anc0, 'range': range_anc0, \
						'seq': anc0_seq0, 'tval': anc0_tvals[0], 'ancn': 0, 'gt': gt_to_anc0, 'tfn': tag['tfn'], 'afn': anc0['afn'], 'dir': tag0_dir}

	if (tag0_valid_anc1 > 0) and (seq_num == anc1_seq0) and (tag0_tvals[0] in tag0_gt):
		anc1 = anc1_data[anc1_tvals[0]]
		fp_poll_anc1 = float((anc1['diag0'])[0])/64.0
		fp_final_anc1 = float((anc1['diag1'])[0])/64.0
		gt_to_anc1 = loc_gt - anc1_loc
		gt_to_anc1 = np.sqrt(gt_to_anc1[0]**2 + gt_to_anc1[1]**2 + gt_to_anc1[2]**2)
		sample['anc1'] = {'poll': anc1['poll'], 'fp_poll': fp_poll_anc1, 'resp': tag['resp_anc1'], 'fp_resp': fp_resp_anc1, 'final': anc1['final'], 'fp_final': fp_final_anc1, 'range': range_anc1, \
						'seq': anc1_seq0, 'tval': anc1_tvals[0], 'ancn': 1, 'gt': gt_to_anc1, 'tfn': tag['tfn'], 'afn': anc1['afn'], 'dir': tag0_dir}

	if (tag0_valid_anc2 > 0) and (seq_num == anc2_seq0) and (tag0_tvals[0] in tag0_gt):
		anc2 = anc2_data[anc2_tvals[0]]
		fp_poll_anc2 = float((anc2['diag0'])[0])/64.0
		fp_final_anc2 = float((anc2['diag1'])[0])/64.0
		gt_to_anc2 = loc_gt - anc2_loc
		gt_to_anc2 = np.sqrt(gt_to_anc2[0]**2 + gt_to_anc2[1]**2 + gt_to_anc2[2]**2)
		sample['anc2'] = {'poll': anc2['poll'], 'fp_poll': fp_poll_anc2, 'resp': tag['resp_anc2'], 'fp_resp': fp_resp_anc2, 'final': anc2['final'], 'fp_final': fp_final_anc2, 'range': range_anc2, \
						'seq': anc2_seq0, 'tval': anc2_tvals[0], 'ancn': 2, 'gt': gt_to_anc2, 'tfn': tag['tfn'], 'afn': anc2['afn'], 'dir': tag0_dir}

	if not sample_empty:
		sample['gt'] = loc_gt
		sample['seq'] = seq_num
		sample['tval'] = tag0_tvals[0]
		sync_data.append(sample)

	tag0_tvals.pop(0)

#Re-alignment and filter out ones that do not have a correctly reported range
new_sync_data = []
for i in range(len(sync_data)-1):
	if ((sync_data[i]['seq'] + 1) % 256) == sync_data[i+1]['seq']:
		if (sync_data[i]['anc0'] is not None) and (sync_data[i+1]['anc0'] is not None):
			sync_data[i]['anc0']['range'] = sync_data[i+1]['anc0']['range']
		else:
			sync_data[i]['anc0'] = None

		if (sync_data[i]['anc1'] is not None) and (sync_data[i+1]['anc1'] is not None):
			sync_data[i]['anc1']['range'] = sync_data[i+1]['anc1']['range']
		else:
			sync_data[i]['anc1'] = None

		if (sync_data[i]['anc2'] is not None) and (sync_data[i+1]['anc2'] is not None):
			sync_data[i]['anc2']['range'] = sync_data[i+1]['anc2']['range']
		else:
			sync_data[i]['anc2'] = None
		new_sync_data.append(sync_data[i])

dataset = {'data': new_sync_data, 'anc0_loc': anc0_loc, 'anc1_loc': anc1_loc, 'anc2_loc': anc2_loc}
print(len(dataset['data']))
with open(f"{test_day}/{traj_id}/{traj_id}.pkl", 'wb') as fp:
	pickle.dump(dataset, fp)

