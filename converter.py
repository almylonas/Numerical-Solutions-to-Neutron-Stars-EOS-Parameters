import csv
with open("bsk20.csv") as f:
    reader = csv.reader(f)
    chunks = []
    for ind, row in enumerate(reader, 1):
        chunks.append(row)
        if ind % 3 == 0: # if we have three new rows, create a file using the first row as the name
            with open("bsk20.csv".format(chunks[0][0].strip(), "w") as f1:
                wr = csv.writer(f1) 
                wr.writerows(chunks) # write all rows
            chunks = [] # reset chunks to an empty list