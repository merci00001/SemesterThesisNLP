import openreview
import requests
import fitz  # PyMuPDF
import pyarrow
import pandas as pd
import pyarrow.parquet as pq

#set the path on where to save the papers
savePath = "/scratch/mgroepl/PaperData/ICLR2023.parquet"

#set the maximal number of papers to check for download. Since even if the paper doesnt exist, it still counts as an api call and can get us rate limited. 
maxPaperNumer = 1000


client = openreview.Client(baseurl='https://api.openreview.net', username='yourOpenReviewUsername', password='YourOpenReviewPassword')

##This block is to find out what naming convention the conference uses. it returns all invitations (in this case conferences) that contain the regex expression. 
##You can use this if you whish to download conferences other than ICLR 2019-2023
invis = openreview.tools.iterget_invitations(client, regex = 'ICLR.cc/')
for y in invis:
    print(y.id)



dfFinal = pd.DataFrame()
paper = 1
while paper <maxPaperNumer:

    ##This invitation schema works for ICLR 2019-2023. If you whish to download other papers, you have to find out the naming convention and change
    ##the invitation variable below to be the final decision of the paper. 
    invitation = "ICLR.cc/2023/Conference/" + "Paper" +str(paper) + "/-/Decision" 
    paper +=1
    notes = openreview.tools.iterget_notes(client, invitation=invitation)
    for note in notes:
        df = pd.DataFrame()


        response = requests.get(
        "https://openreview.net/pdf?id=" + note.forum,
        headers={},
        )

        pdf_filename = "downloaded.pdf"
        txt_filename = "output.txt"

        if response.status_code == 200:
            with open(pdf_filename, "wb") as f:
                f.write(response.content)

            # Convert PDF to text
            doc = fitz.open(pdf_filename)
            text = "\n".join(page.get_text() for page in doc)

            df["Paper"] = [text]
         
            # Save to text file
            #with open(txt_filename, "w", encoding="utf-8") as f:
            #    f.write(text)

            #print("PDF downloaded and converted to text successfully!")
        else:
            print("Failed to get PDF")
            continue
        df["decision"] = [note.content["decision"]]
        print(paper)

        notesForum = openreview.tools.iterget_notes(client, forum=note.forum)
        amnt = 0
        totalRating = 0
        totalAmntRatings = 0
        for rep in notesForum:
            dicts = {}
            #Skip the paper 
            if rep.id ==rep.forum:
                continue
            
            #Skip replies to replies
            if rep.forum != rep.replyto:
                continue
            
            if "review" not in rep.content:
                continue

            dicts["review"] = rep.content["review"]

            dicts["rating"] = rep.content["rating"]
            totalRating += int(rep.content["rating"][0])
            totalAmntRatings +=1

            #dicts["confidence"] = rep.content["confidence"]
            df[str(amnt)] = [dicts]
            amnt += 1
        if totalAmntRatings != 0:
            df["mean"] = [totalRating / totalAmntRatings]
        else:
            df["mean"] = [0]
        dfFinal = pd.concat([dfFinal,df],ignore_index = True)
                



table = pyarrow.Table.from_pandas(dfFinal)
print(table.shape)
pq.write_table(table,"/scratch/mgroepl/PaperData/ICLR2023.parquet")

