# API FEEDBACK
    # ------------
    #
    # Why is the API different from Python?
    #   x = b"123"
    #   y = x.decode("utf-8")
    #
    # I'd expect the API to be:
    #   df: DataFrame = ...
    #   df.with_column("text", col("warc_content").str.decode("utf-8"))
    # Or even:
    #   df.with_column("text", col("warc_content").binary.decode("utf-8"))
    #
    # But instead decode is a top-level method on **any** expression?
    # The expression should have to be in the binary namespace!
    #   df = df.with_column("text", col("warc_content").decode("utf-8"))
