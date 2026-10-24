"""Hand-written topic-oriented queries for the public corpora.

Relevance is proxied by category membership: every document whose category
equals the query's category is judged relevant (proposal section 4.3).
Queries are deliberately short and in natural vocabulary, not copied from
documents, so that they behave like ad-hoc user queries.
"""

NEWSGROUPS_QUERIES = [
    ("alt.atheism", "arguments against existence of god atheism"),
    ("alt.atheism", "atheist morality without religion"),
    ("comp.graphics", "3d graphics rendering software image formats"),
    ("comp.graphics", "convert gif jpeg image file"),
    ("comp.os.ms-windows.misc", "windows 3.1 crash driver problem"),
    ("comp.sys.ibm.pc.hardware", "ide scsi hard drive controller motherboard"),
    ("comp.sys.ibm.pc.hardware", "pc bios memory upgrade jumper"),
    ("comp.sys.mac.hardware", "apple macintosh hardware monitor upgrade"),
    ("comp.windows.x", "x window system motif xterm server"),
    ("misc.forsale", "for sale price shipping offer condition"),
    ("rec.autos", "car engine dealer new model price"),
    ("rec.autos", "tires brakes oil change car maintenance"),
    ("rec.motorcycles", "motorcycle riding helmet bike"),
    ("rec.sport.baseball", "baseball pitcher hitting season game"),
    ("rec.sport.baseball", "home runs batting average players"),
    ("rec.sport.hockey", "hockey nhl playoffs goalie team"),
    ("sci.crypt", "encryption key clipper chip privacy"),
    ("sci.crypt", "public key cryptography algorithm security"),
    ("sci.electronics", "circuit voltage power supply electronics"),
    ("sci.med", "disease treatment doctor patients symptoms"),
    ("sci.med", "diet food medical research cancer"),
    ("sci.space", "nasa space shuttle launch orbit"),
    ("sci.space", "moon mission satellite spacecraft"),
    ("soc.religion.christian", "christian church bible faith jesus"),
    ("talk.politics.guns", "gun control firearms law weapons"),
    ("talk.politics.mideast", "israel arab palestinian conflict"),
    ("talk.politics.mideast", "armenian turkish genocide history"),
    ("talk.politics.misc", "government taxes clinton policy"),
    ("talk.religion.misc", "religion morality christian beliefs debate"),
    ("comp.os.ms-windows.misc", "windows file manager program install"),
]

REUTERS_TOPICS = ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship", "sugar", "coffee",
                  "gold", "money-supply", "gnp", "cpi", "cocoa", "copper", "jobs", "iron-steel",
                  "grain", "alum", "nat-gas"]

REUTERS_QUERIES = [
    ("earn", "quarterly net profit earnings per share"),
    ("earn", "company reports dividend revenue results"),
    ("acq", "company acquisition merger takeover bid"),
    ("acq", "agreed to buy stake shares acquire"),
    ("crude", "crude oil prices opec production"),
    ("crude", "oil barrels output refinery"),
    ("trade", "trade deficit tariffs imports exports"),
    ("trade", "trade dispute japan united states"),
    ("money-fx", "currency exchange rates dollar intervention"),
    ("interest", "interest rate cut central bank prime rate"),
    ("ship", "shipping port vessels strike cargo"),
    ("sugar", "sugar crop production tonnes"),
    ("coffee", "coffee export quotas ico prices"),
    ("gold", "gold mine ounces production"),
    ("money-supply", "money supply m1 m3 growth"),
    ("gnp", "gross national product economic growth forecast"),
    ("cpi", "consumer price index inflation rate"),
    ("cocoa", "cocoa buffer stock prices"),
    ("copper", "copper smelter mine prices"),
    ("jobs", "unemployment rate jobless workers"),
    ("iron-steel", "steel production mills iron ore"),
    ("grain", "wheat corn grain harvest"),
    ("alum", "aluminium smelter production prices"),
    ("nat-gas", "natural gas pipeline reserves"),
    ("money-fx", "yen mark sterling foreign exchange markets"),
]
