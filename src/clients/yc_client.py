import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json


class YCClient:
    def __init__(self, headless=True):
        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")

        self.service = Service(ChromeDriverManager().install())
        self.driver = None

    def _initialize_driver(self):
        if self.driver is None:
            self.driver = webdriver.Chrome(service=self.service, options=self.options)
        return self.driver

    def _render_page(self, pause_time=2):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _load_batch_page(self, batch_code):
        base_url = "https://www.ycombinator.com/companies"
        params = []

        if batch_code != "top_company":
            params.append(f"batch={batch_code}")
        else:
            params.append("top_company=true")

        url = f"{base_url}{'?' + '&'.join(params) if params else ''}"
        self._initialize_driver()
        self.driver.get(url)
        time.sleep(4)
        self._render_page(pause_time=2)

    def get_company_urls_for_batch(self, batch_code):
        self._load_batch_page(batch_code)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        company_list = []
        # The company cards are <a class="_company_i9oky_355">
        cards = soup.find_all("a", class_="_company_i9oky_355")
        for card in cards:
            href = card.get("href", "")
            if href.startswith("/companies/"):
                name_tag = card.find("span", class_="_coName_i9oky_470")
                company_name = name_tag.get_text(strip=True) if name_tag else "Unknown"
                company_url = "https://www.ycombinator.com" + href
                company_list.append((company_name, company_url))
        return company_list

    def scrape_founders_from_company(self, company_name, company_url, batch_code):
        founders_data = []
        try:
            self._initialize_driver()
            self.driver.get(company_url)
            time.sleep(1)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            container = soup.find(
                "div", id=lambda i: i and i.startswith("ycdc_new/pages/Companies/ShowPage-react-component")
            )
            if not container:
                founders_data.append({"Name": None, "Company": company_name, "LinkedIn": None, "Batch": batch_code})
                return founders_data

            # Pull out the JSON
            raw_json = container.get("data-page", "")
            if not raw_json:
                founders_data.append({"Name": None, "Company": company_name, "LinkedIn": None, "Batch": batch_code})
                return founders_data

            parsed = json.loads(raw_json)
            # props->company->founders
            founders = parsed["props"]["company"].get("founders", [])

            # Build records
            for f in founders:
                founders_data.append(
                    {
                        "Name": f.get("full_name"),
                        "Company": company_name,
                        "LinkedIn": f.get("linkedin_url"),
                        "Batch": batch_code,
                    }
                )

            # fallback
            if not founders_data:
                founders_data.append({"Name": None, "Company": company_name, "LinkedIn": None, "Batch": batch_code})

        except Exception as e:
            print(f"Error processing {company_name} => {e}")
            founders_data.append({"Name": None, "Company": company_name, "LinkedIn": None, "Batch": batch_code})

        return founders_data

    def scrape_batch(self, batch_code, sleep_time=0.5):
        self._initialize_driver()

        all_records = []
        company_list = self.get_company_urls_for_batch(batch_code)
        print(f"Found {len(company_list)} companies for batch {batch_code}.")

        for i, (company_name, company_url) in enumerate(company_list):
            print(f"Scraping {(float(i)/len(company_list)*100):.2f}%: {company_name}: {company_url}")
            records = self.scrape_founders_from_company(company_name, company_url, batch_code)
            all_records.extend(records)

            time.sleep(sleep_time)

        df = pd.DataFrame(all_records, columns=["Name", "Company", "LinkedIn", "Batch"])
        return df

    def scrape_batches(self, batch_codes, output_dir=None):
        self._initialize_driver()

        all_data = pd.DataFrame()

        for batch_code in batch_codes:
            print(f"Scraping YC companies for batch: {batch_code}")
            batch_df = self.scrape_batch(batch_code)
            all_data = pd.concat([all_data, batch_df], ignore_index=True)

            if output_dir:
                batch_df.to_csv(f"{output_dir}/{batch_code}.csv", index=False)

        return all_data

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def __del__(self):
        self.close()
