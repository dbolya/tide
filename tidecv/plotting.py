def print_table(rows: list, title: str = None):
    # Get all rows to have the same number of columns
    max_cols = max([len(row) for row in rows])
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    # Compute the text width of each column
    col_widths = [
        max([len(rows[i][col_idx]) for i in range(len(rows))])
        for col_idx in range(len(rows[0]))
    ]

    divider = "--" + ("---".join(["-" * w for w in col_widths])) + "-"
    thick_divider = divider.replace("-", "=")

    if title:
        left_pad = (len(divider) - len(title)) // 2
        print(("{:>%ds}" % (left_pad + len(title))).format(title))

    print(thick_divider)
    for row in rows:
        # Print each row while padding to each column's text width
        print(
            "  "
            + "   ".join(
                [
                    ("{:>%ds}" % col_widths[col_idx]).format(row[col_idx])
                    for col_idx in range(len(row))
                ]
            )
            + "  "
        )
        if row == rows[0]:
            print(divider)
    print(thick_divider)
