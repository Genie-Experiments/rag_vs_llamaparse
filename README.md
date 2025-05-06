# rag_vs_llamaparse
This repo will contain an experiment that compares a regular rag pipeline vs one that uses llamaparse
to start, download all the requirements from requirements.txt and 
setup the .env file with the LLAMA_CLOUD_API_KEY and the GROQ_API_KEY.

To setup the regular rag pipeline, create a "docs" folder in the project and add the documents you need to 
ingest to this folder.
Once done, run the ingest.py file first and then run the rag_pipeline.py file.

To setup the llamaparse pipeline, you need to add the paths to the documents in the docs folder in the list 
files_to_load right under the comment "PDF files" in the llamaparse_pipeline.py file. Then run the file. 

The results from both the files will be stored in the llamaparse_dataset.json file.

Note: Currently, the questions are being fetched from the llamaparse_dataset.json file, so for the scripts to work, you
will have to create this file and add the questions in the following format:

[
{"question": "",
"ground_truth": "",
"type": ""
}
]

The code will add the responses from each pipeline to this json.