# Ojas_EB

# Structured_chunker3.py
-this is for Blog, can test individual websites

# test_chunker.py
for those websites whose chunks are less than 4 to test them and check their output.

# curated_chunker1.py
does what test_chunker does but takes a csv file as input where urls to crawl are present in the first column.

# trial_nollm_crawl1.py
for about dementia and get support

# curated_chunker_with_log.py
for blog this is default and can be used to do structured_chunker3.py on large scale with a log file if chunk number is less than 4 or more than 7

# curated_chunker_with_log1.py
does what the curated_chunker_with_log.py does but discards the chunks which doesn't fall in 4-7 range.

# refine_chunks3.py
to refine chunks into 300 to 600

# chunk_and_upload.py
to upload to vector database.

# chatbot_agent_executor6.py
the agent file - run this to test agent and vector database

# test_retrieval.py
tests and retrives relavent info from vector database without doing llm tool call




