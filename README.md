# DeepCIR
DeepCIR: Insights into CIR-based Data-driven UWB Error Mitigation for Indoor Navigation

Please be patient. A clean version of dataset and code will be uploaded soon!

Updated: the extracted dataset has been uploaded (in dataset/train) which is ready for training. Each data file contains CIR buffer (120-sample), each label file contains the error in metre. Please note that the "double" means double-sided CIR. train_cir_poll.npy, train_cir_resp.npy and train_cir_final.npy are synchronized (same for val-). It means that train_cir_poll[index],train_cir_resp[index],train_cir_final[index] belong to the same transaction. The FMCIR and WMCIR models as well as the baseline models are in code/models.py

The raw dataset is too big, please be patient while the raw dataset is simplified (but still contains essential details such as CIR, raw FPI, timestamps)! Hopefully the entire raw dataset and code will be released by Sun August 7th 2022.

If you have any request, please contact Vu Tran at vu.tran.apollo@gmail.com or vu.tran@cs.ox.ac.uk 

Thank you.