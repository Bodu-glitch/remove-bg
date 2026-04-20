import shutil
import browser_agent as m

m.MAX_IMAGES = 2
shutil.rmtree(m.OUT, ignore_errors=True)
m.OUT.mkdir()
shutil.rmtree(m.SCREENSHOTS, ignore_errors=True)
m.SCREENSHOTS.mkdir()
m.main()
