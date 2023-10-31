import json
AUC = []
recall_1 = []
recall_3 = []
recall_5 = []
ppl = []
with open("GP_tr_p3_lora_all_Books.log") as f:
    for line in f:
        if line.startswith("{'AUC': "):
            line = eval(line)
            AUC.append(line["AUC"])
        elif line.startswith("{'recall@1': "):
            line = eval(line)
            recall_1.append(line["recall@1"])
            recall_3.append(line["recall@3"])
            recall_5.append(line["recall@5"])
        elif line.startswith("{'ppl': "):
            line = eval(line)
            ppl.append(line["ppl"])
for x in recall_5[1::2]:
    print(x)