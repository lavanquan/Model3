import csv
f = open("ketqua/haha.csv", mode="w")
header = ["Bo Du Lieu", "Thoi Gian Song"]
writer = csv.DictWriter(f, fieldnames=header)
writer.writeheader()
for i in range(5):
    row = {}
    row["Bo Du Lieu"] = "No. " + str(i+1)
    row["Thoi Gian Song"] = i + 1
    writer.writerow(row)
f.close()