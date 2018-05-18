

DATA_ROOT = r"E:\Github2\ML-Conation\DataParser\\"

def load_single_data(path):

    with open(path, "r") as ins:
        array = list()
        for line in ins:
            array.append(line)

    return array

def prep_data(filename):
    data = load_single_data(filename)

    header = data[0]
    data = data[1:]

    conation = list()
    for x in range(len(data)):
        conation.append(data[x][-4:-1])
        data[x] = data[x][:-5]

    print(conation[123])

    a = list()
    b = list()
    d = list()

   # delimiters = list()
    indices = list()

    for x in range(len(data)):
        if (data[x].count(',') == 17):
            a.append(data[x])
        if (data[x].count(',') == 18):
            b.append(data[x])
        if (data[x].count(',') == 19):
            d.append(data[x])
            indices.append(x)


    print(len(d))
    print(len(indices))

    split = [[" " for i in range(20)] for j in range(len(d))]
    gathered = [[" " for i in range(10)] for j in range(len(d))]
    sentences = [[" " for i in range(10)] for j in range(len(d)+1)]

    sentences[0] = header
    print(sentences[0])
    for x in range(len(d)):
        split[x] = d[x].split(',')



# gathering file
    for x in range(len(d)):
        for y in range(0, 10):
            gathered[x][y] = '.'.join( [split[x][y*2],split[x][y*2+1]])


        gathered[x].append(',')
        gathered[x].append(conation[indices[x]])

        # 8 heart rate
        tempHR = str(round(float(  gathered[x][8])/100  ,2))
        gathered[x][8] = tempHR



        # 9 gsr
        tempGSR = gathered[x][9]
        if tempGSR[0] != '-':

            if tempGSR[-2:] == "16":
                tempGSR = "0.0" + tempGSR[0] + tempGSR[2:-4]


            if tempGSR[-2:] == "15":
                tempGSR = "0.00" +  tempGSR[0] + tempGSR[2:-4]

        if tempGSR[0] == '-':

            if tempGSR[-2:] == "16":
                tempGSR = "-0.0" + tempGSR[1] + tempGSR[3:-4]


            if tempGSR[-2:] == "15":
                tempGSR = "-0.00" +  tempGSR[1] + tempGSR[3:-4]

        gathered[x][9]  =  tempGSR


           # gathered[x][9] = "0." +gathered[x][9][0] + gathered[x][9][2:9]

       # 3,53219216473405E+16
       # -4,76493701582767E+15

       # 0.023842255633017
       # -0.00236896654832308



        sentence = ""
        for z in range(1,len(gathered[x])):

            sentence +=gathered[x][z]
            if(z<9):
                sentence += ','

        if x != 0:
            sentences[x] = sentence
    print(sentences[0])

    #print(sentences[3])


    return  sentences

def write_to_file(array, filename):
    file = open(filename,"w")

    for x in range(len(array)-2):

        file.write(array[x])
        file.writelines("\n")

    file.close()

def find(str, ch):
    for i, ltr in enumerate(str):
        if ltr == ch:
            yield i


#prep_data('Data01.txt')
#write_to_file(prep_data('Data02.txt'), "newFile1.txt")




filebase = "Data0"
newFileName = "CorrectedData0"
enum = 1
for x in range(0,9):
    #prep_data(filebase + str(enum) + ".txt")
    write_to_file(prep_data(filebase + str(enum) + ".txt"), newFileName+ str(enum) + ".txt")
    enum +=1








# 2O KOMMAER  10 værdier + 2 conation