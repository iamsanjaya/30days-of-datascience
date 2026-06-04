# Generates a synthetic book with emotional variation across chapters

chapters = [
    (
        "Chapter I",
        """
The morning was bright and full of hope. The children laughed and played in the
warm sunshine. Everyone felt joy and happiness filled the air with wonderful
sounds. Life was beautiful and friends gathered in delight. Sweet dreams and
peace surrounded the gentle village. Love and warmth touched every soul.
""",
    ),
    (
        "Chapter II",
        """
Suddenly a dark shadow fell upon the land. Fear and danger crept through the
streets. The people ran in desperate silence, hiding from the terrible threat.
Blood and horror filled the night. Death came swiftly and pain echoed through
the cold darkness. Cruel and wicked forces attacked without warning.
""",
    ),
    (
        "Chapter III",
        """
Hope returned slowly. A kind stranger brought warmth and gentle words. The
wounded began to heal and grateful hearts found peace again. Beautiful mornings
returned and children smiled once more. Love triumphed and bright days followed
the long darkness. Friends reunited with joy and sweet laughter.
""",
    ),
    (
        "Chapter IV",
        """
The enemy struck again with sudden fury. Desperate citizens fought back in
urgent chaos. The chase through dark streets filled everyone with alarm and
shock. Strange and unknown dangers lurked in every shadow. Silent watchers
waited with terrible patience. The risk of death hung over every soul.
""",
    ),
    (
        "Chapter V",
        """
Victory came at last. Glory and triumph filled the hearts of the brave.
Wonderful celebrations brought joy and laughter to every corner. Peace and
love returned to the land. Grateful people danced with delight under the
bright warm sun. Life was beautiful once more and hope never died.
""",
    ),
]

with open("day-02/data/book.txt", "w") as f:
    for title, content in chapters:
        f.write(f"\n{title}\n")
        f.write(content * 10)  # repeat to give enough word count

print("Generated day-02/data/book.txt successfully")
