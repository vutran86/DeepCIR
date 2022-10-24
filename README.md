# DeepCIR
DeepCIR: Insights into CIR-based Data-driven UWB Error Mitigation for Indoor Navigation

Please be patient. A clean version of dataset and code will be uploaded soon!

Updated: the extracted dataset has been uploaded (in dataset/train) which is ready for training. Each data file contains CIR buffer (120-sample), each label file contains the error in metre. Please note that the "double" means double-sided CIR. train_cir_poll.npy, train_cir_resp.npy and train_cir_final.npy are synchronized (same for val-). It means that train_cir_poll[index],train_cir_resp[index],train_cir_final[index] belong to the same transaction. The FMCIR and WMCIR models as well as the baseline models are in code/models.py

Please download the raw dataset from my googledrive (it is too big for github) using [this link](https://drive.google.com/file/d/1YCYWVyXA_92Huwdrvyt7Udk4TJyofzZW/view?usp=sharing). Extract all the content into dataset/raw. You will see something like this:
dataset
  + raw
    + combine.py
    + syncSessions.py
    + syncTrjectory.py
    + dec22
    + feb04
    + ...
  + test
  + train

Please run syncSessions.py dec22 sess1 2021-12-22-15-59 to extract the data for dec22 session1. The timestamp 2021-12-22-15-59 can be found at dec22/sess1/anc1/log-anc0-2021-12-22-15-59 where many UWB transactions are stored (together with anc1/anc2/tag0). The script then synchronizes each transaction using the fine-grained timestamp stored in each file.

The first 48 bytes in each file can be unpacked as meta data "QBBBBIIIIIIIII" where meta[0] is the record timestamp which is an unsinged long integer (8-bytes) at the beginning of each file (micro second unit). meta[5,6,7] is the distance from the tag to anc0, anc1, anc2. Please refer to the helper code syncSessions.py and combine.py for other details such as sequence number, first_path_index (FPI) or reception & transmission timestamp of each packet in a transaction. (Or wait for a while -- I will write more when I have time. Thanks for your patience).

If you have any request, please contact Vu Tran at vu.tran.apollo@gmail.com (Please do not send email to vu.tran@cs.ox.ac.uk as I have left the university) 

Thank you.
