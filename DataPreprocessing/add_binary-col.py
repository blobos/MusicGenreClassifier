#NOT USED
#for testing usiing binary classification
#appends if genre is selected genre or not on csv
import csv
og_csv = "/home/student/Music/1/FYP/data/train_annotations.csv"
bin_csv = "/home/student/Music/1/FYP/data/train_annotations_binary.csv"
# with open(csv_path, "a") as csv_file:
#         reader = csv.reader(csv_file)
#         headers = next(reader)
#         headers.append('punk')
#
#         writer = csv.writer(csv_file)
#         writer.writerow(headers)


with open(og_csv, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    # print(headers)
    headers.append('punk')
    # print(headers)

    with open(bin_csv, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        for row in reader:
            if row[3] == '9':
                row.append('1')
                print(row)
            else:
                row.append('0')
                # print(row)
            writer.writerow(row)

