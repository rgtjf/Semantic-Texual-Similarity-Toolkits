import time, os
import traceback

import lxml.html
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class AsiyaDriver(object):
    def __init__(self):
        driver = webdriver.Chrome()
        driver.get("http://asiya.cs.upc.edu/demo/asiya_online.php#")
        time.sleep(3)
        driver._switch_to.frame(driver.find_element_by_id("demo-content"))
        elem = Select(driver.find_element_by_id("input"))
        elem.select_by_value("raw")
        elem = driver.find_element_by_id("no_tok")
        elem.click()
        elem = Select(driver.find_element_by_id("srclang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("trglang"))
        elem.select_by_value("en")
        elem = Select(driver.find_element_by_id("srccase"))
        elem.select_by_value("ci")
        elem = Select(driver.find_element_by_id("trgcase"))
        elem.select_by_value("ci")

        self.driver = driver

    def run(self, sa, sb):
        driver = self.driver
        try:

            elem = driver.find_element_by_id("srctext")
            elem.send_keys(sa)

            elem = driver.find_element_by_id("systext")
            elem.send_keys(sb)

            elem = driver.find_element_by_id("reftext")
            elem.send_keys(sb)

            elem = driver.find_element_by_id("submitbtndoasiya")
            elem.click()

            elem = WebDriverWait(driver, 300).until(
                EC.presence_of_element_located((By.ID, "div_scores"))
            )
            headers = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//thead//tr//th")

            rows = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//tbody//tr//td")

            rows = [ eval(row.text) for row in rows[3:]]

            # BLEU, GTM - 3, NIST, -WER, -PER, Ol, -TERbase, METEOR - ex, ROUGE - L

            elem = driver.find_element_by_xpath("//input[@value='Clear Files']")
            elem.click()
        except:
            traceback.print_exc()
            rows = []
        return rows

    def run_file(self, src='sa.txt', sys='sb.txt'):
        ''' write to file sa.txt, sb.txt'''

        driver = self.driver

        try:
            elem = driver.find_element_by_id("srcupload")
            elem.click()
            time.sleep(3)
            os.system("upload_sa.exe")
            time.sleep(3)


            elem = driver.find_element_by_id("sysupload")
            elem.click()
            time.sleep(3)
            os.system("upload_sb.exe")
            time.sleep(3)

            elem = driver.find_element_by_id("refupload")
            elem.click()
            time.sleep(3)
            os.system("upload_sa.exe")
            time.sleep(3)

            elem = driver.find_element_by_id("submitbtndoasiya")
            elem.click()

            elem = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "div_scores"))
            )

            page = driver.page_source
            # headers = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//thead//tr//th")

            # rows = driver.find_elements_by_xpath("//table[@id='tab_scores_mmatrix_grseg']//tbody//tr//td")
            # rows = [eval(row.text) for row in rows[3:]]

            # BLEU, GTM - 3, NIST, -WER, -PER, Ol, -TERbase, METEOR - ex, ROUGE - L

            elem = driver.find_element_by_xpath("//input[@value='Clear Files']")
            elem.click()
        except:
            traceback.print_exc()
            # rows = []
            page = ''
        return page


    def extract_table(self, page_source):
        root = lxml.html.fromstring(page_source)
        for row in root.xpath('.//table[@id="tab_scores_mmatrix_grseg"]//tbody//tr'):
            cells = row.xpath('.//td/text()')
            dict_value = {'0th': cells[0],
                          '1st': cells[1],
                          '2nd': cells[2],
                          '3rd': cells[3],
                          '6th': cells[6],
                          '7th': cells[7]}
            print(dict_value)

    def quit(self):
        self.driver.close()


asiya = AsiyaDriver()
page = asiya.run_file()
asiya.extract_table(page)
asiya.quit()
# import config, data_utils
# train_file = config.TRAIN_FILE
# train_gs = config.TRAIN_GS_FILE
# train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)
#
# f_sa = open(config.EX_DICT_DIR + '/sa.txt', 'w')
# f_sb = open(config.EX_DICT_DIR + '/sb.txt', 'w')
# print()
# for idx, train_instance in enumerate(train_parse_data):
#     lemma_sa, lemma_sb = train_instance.get_word(type='lemma')
#     lemma_sa = ' '.join(lemma_sa)
#     lemma_sb = ' '.join(lemma_sb)
#     print(lemma_sa, file=f_sa)
#     print(lemma_sb, file=f_sb)
#     if idx - 400 == 0:
#         break
