# DeepCIR
DeepCIR: Insights into CIR-based Data-driven UWB Error Mitigation for Indoor Navigation

Please be patient. A clean version of dataset and code will be uploaded soon!

Updated: the extracted dataset has been uploaded (in dataset/train) which is ready for training. Each data file contains CIR buffer (120-sample), each label file contains the error in metre.

The convention in the double label file is Error(metre), Estimated distance(metre -- raw value estimated by the sensor), Groundtruth distance(metre), Poll FP index, Resp FP index, Final FP index

The convention in the single label file is Error(metre), Estimated distance(metre -- raw value estimated by the sensor), Groundtruth distance(metre), FP index

Please note that the "double" means double-sided CIR. train_cir_poll.npy, train_cir_resp.npy and train_cir_final.npy are synchronized (same for val-). It means that train_cir_poll[index],train_cir_resp[index],train_cir_final[index] belong to the same transaction. The FMCIR and WMCIR models as well as the baseline models are in code/models.py

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

The first 48 bytes in each file can be unpacked as meta data "QBBBBIIIIIIIII" where meta[0] is the record timestamp which is an unsinged long integer (8-bytes) at the beginning of each file (micro second unit). meta[1] is node address: bit_7 == 0 indicates anchor, 3 LSB bits indicate node index (anchor/tag 0,1,2,...). meta[2] indicates if the measurement is valid or not. meta[5,6,7,8] are the distances from the tag to anc0, anc1, anc2 (and anc3 if used). meta[9,10,11,12] are the corresponding raw measurements.

The next 30 bytes (start at 48) are 6 timestamps recorded in a transaction, each 5 bytes: Poll_Tx, Final_Tx, Resp_Tx, Final_Rx, Poll_Rx, Resp_Rx. The timestamps are recorded at the anchors only, the tag also reports its timestamps to the anchors. For example, anchor 0 recorded all timestamps between the tag and anchor 0 (6 timestamps); anchor 1 recorded all timestamps between the tag and anchor 1 (6 timestamps).

The next 17 bytes (start at 78) are 8 16-bit fields and 1 8-bit field: FPI (First Path Index) of the packet, FPIAMp1, FPIAmp2, FPIAmp3, MaxGrowCIR, MaxNoise, StdNoise, RxPreamCount, and Sequence Number (8-bit). Please read DWM1000 datasheet & User manual for the details of those diagnostic values.

** ! Note that the sensor (Poll) reports the range for the previous transaction (previous sequence number) **.

The CIR data starts at 98 (78 + 20), including 120 samples (16-bit real, 16-bit imaginary).

If you have any request, please contact Vu Tran at vu.tran.apollo@gmail.com (Please do not send email to vu.tran@cs.ox.ac.uk as I have left the university) 

Thank you.
