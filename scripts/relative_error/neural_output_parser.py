#!/usr/bin/env python
import os
import re
import glob
import numpy as np

header = "#!/usr/bin/env python\n\
\n\
import sys\n\
import os\n\
import csv\n\
import re\n\
import collections\n\
import parseUtils\n\
\n\
\n\
def processErrors(errList):\n\
        errLimitList = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,8,10,12,15]\n\
        csvHeader = list()\n\
        csvHeader.append(\"numberIterationErrors\")\n\
        csvHeader.append(\"maxRelativeError\")\n\
        csvHeader.append(\"minRelativeError\")\n\
        csvHeader.append(\"averageRelativeError\")\n\
        for errLimit in errLimitList:\n\
            csvHeader.append(\"numberErrorsLessThan\"+str(errLimit))\n\
\n\
\n\
        csvOutDict = dict()\n\
        csvOutDict[\"numberIterationErrors\"]=len(errList)\n\
        for errLimit in errLimitList:\n\
            (maxRelErr, minRelErr, avgRelErr, relErrLowerLimit, errListFiltered) = parseUtils.relativeErrorParser(errList, errLimit)\n\
            csvOutDict[\"maxRelativeError\"] = maxRelErr\n\
            csvOutDict[\"minRelativeError\"] = minRelErr\n\
            csvOutDict[\"averageRelativeError\"] = avgRelErr\n\
            csvOutDict[\"numberErrorsLessThan\"+str(errLimit)] = relErrLowerLimit\n\
\n\
\n\
	# Write info to csv file\n\
        csvFullPath = \"out.csv\"\n\
\n\
        if not os.path.isfile(csvFullPath):\n\
	    csvWFP = open(csvFullPath, \"a\")\n\
	    writer = csv.writer(csvWFP, delimiter=';')\n\
            writer.writerow(csvHeader)\n\
        else:\n\
	    csvWFP = open(csvFullPath, \"a\")\n\
	    writer = csv.writer(csvWFP, delimiter=';')\n\
        row = list()\n\
        for item in csvHeader:\n\
            if item in csvOutDict:\n\
                row.append(csvOutDict[item])\n\
            else:\n\
                row.append(\" \")\n\
        writer.writerow(row)\n\
\n\
	csvWFP.close()\n\
\n\
\n\
###########################################\n\
# MAIN\n\
###########################################\n\
csvDirOut = \"csv_logs_parsed\"\n\
print \"\\n\\tCSV files will be stored in \"+csvDirOut+\" folder\\n\"\n\n"

process_list = "### Process errors\n"

gold = [0.000144644,
	0.000463701,
	0.999867,
	0.999616,
	9.443e-05,
	0.00028905,
	0.99986,
	0.999814,
	0.000115164,
	0.000176446,
	0.999873,
	0.999853]

cwd = os.getcwd()
current_folder_path, current_folder_name = os.path.split(os.getcwd())
print ("Processing directory "+cwd)
print ("Processing folder "+current_folder_name)

process_file = open("process.py", "w")
process_file.write(header)

i = 1
for f in glob.glob('../../logs/*/sdcs*/*/*/output'):
    with open(f) as output:
        print ("Processing file " + f)
        results = np.genfromtxt(f)
        if results.__len__() == gold.__len__():
            process_file.write("#### Example errors from sdc #" + str(i) +"\n" +
                               "#### File: " + f + "\n" +
                               "errList" + str(i) + " = list()\n\n")
            for x in range(0, gold.__len__()):
                if gold[x] != results[x]:
                    process_file.write("positionsExample = [[\"x\", " + str(x) + "]]\n" +
                                       "valuesExample=[[\"value\", " + str(gold[x]) +", " + str(results[x]) + "]]\n")

                    process_file.write("errItem = {\"position\" : positionsExample, \"values\" : valuesExample}\n" +
                                       "errList" + str(i) + ".append(errItem)\n\n")
            process_list += "processErrors(errList" + str(i) + ")\n"

            i += 1

process_file.write(process_list)
process_file.close()
    
