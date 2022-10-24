import pickle
import numpy as np 
import matplotlib.pyplot as plt
import glob
import sys 

test_date = sys.argv[1]
if len(sys.argv) > 2:
    sess_id = sys.argv[2]
    sess_list = [f"{test_date}/{sess_id}"]
else:
    sess_list = glob.glob(f"{test_date}/sess*")
combined_dataset = []
for sess in sess_list:
    sess_id = sess.split("/")[1]
    filename = f"{sess}/{sess_id}.pkl"
    with open (filename, 'rb') as fp:
        dataset = pickle.load(fp)
        sync_data = dataset['data']
        for k in range(len(sync_data)):
            if sync_data[k]['anc0'] is not None:
                combined_dataset.append(sync_data[k]['anc0'])
                
            if sync_data[k]['anc1'] is not None:
                combined_dataset.append(sync_data[k]['anc1'])
                
            if sync_data[k]['anc2'] is not None:
                combined_dataset.append(sync_data[k]['anc2'])

error = []
for i in range(len(combined_dataset)):
    error.append(combined_dataset[i]['range']/1000.0 - combined_dataset[i]['gt'])
print(len(combined_dataset))
with open(f"{test_date}/{test_date}.pkl", 'wb') as fp:
    pickle.dump(combined_dataset, fp)

print(np.mean(np.abs(error)))
plt.hist(error, bins=100)
plt.show()
