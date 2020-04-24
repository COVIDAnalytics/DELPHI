require(tidyverse)
require(data.table)
setwd("/Users/Ted/Documents/mit/research/DELPHI/data_sandbox")

raw_dens = fread("Current_Population_Density_RAND_US_1.csv")
raw_pop = fread("Current_Population_Estimates_RAND_US_1.csv")
raw_cases = fread("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
raw_mobil = fread("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv", colClasses = "character")

POP_YEAR = 2018        # What year's population and density to use from RAND
POP_CUTOFF = 100000    # Minimum population for a county to be chosen
PICK_N = 50            # How many counties to select
PICK_VAR = "CurCases"  # Which variable to sort and pick top PICK_N off
fout = "Top50LargeCounties_0423.csv" # output file

# Remove duplicate density rows, select appropriate year column, convert FIPS to number
dens = raw_dens %>% 
    filter(!duplicated(`FIPS Code`)) %>% 
    select(State, Area, FIPS = `FIPS Code`, 
           Density = !!rlang::sym(as.character(POP_YEAR))) %>% 
    mutate(FIPS = as.numeric(FIPS))

# Remove duplicate population rows, select appropriate year column, convert FIPS to number
pop = raw_pop %>% 
    filter(!duplicated(`FIPS Code`)) %>% 
    select(FIPS = `FIPS Code`, Pop = !!rlang::sym(as.character(POP_YEAR))) %>% 
    mutate(FIPS = as.numeric(FIPS))


# Filter to where we have FIPS code, pick most recent date for each FIPS code
cases = raw_cases %>% 
    filter(!is.na(fips)) %>% 
    group_by(fips) %>% 
    filter(min_rank(desc(date)) == 1) %>% # Pick most recent date in each FIPS
    ungroup %>%
    select(State = state, Area = county, FIPS = fips, 
           CurCases = cases, CurDeaths = deaths, CurDate = date) %>% 
    mutate(FIPS = as.numeric(FIPS))

# Join everything together. Left join onto cases, only want FIPS with any cases.
# Compute Cases and Deaths per 1000 people
df = cases %>% 
    select(-State, -Area) %>% # I'll take these from the RAND population data, seems to better much names in mobility file
    left_join(dens, by = "FIPS") %>% 
    left_join(pop, by = "FIPS") %>% 
    filter(!is.na(Pop), !is.na(Density)) %>% 
    mutate(CurCasesRate = CurCases / (Pop/1000),
           CurDeathsRate = CurDeaths / (Pop/1000)) %>% 
    select(FIPS, State, County = Area, everything())

mobil = raw_mobil %>% 
    # Filter mobility down to counties
    filter(country_region_code == "US" & sub_region_1 != "" & sub_region_2 != "" | 
           country_region_code == "US" & sub_region_1 == "District of Columbia") %>% 
    # Compute whether we have everything for the row
    mutate(IsCompleteRow = 
               retail_and_recreation_percent_change_from_baseline != "" &
               grocery_and_pharmacy_percent_change_from_baseline != "" & 
               parks_percent_change_from_baseline != "" &                
               transit_stations_percent_change_from_baseline != "" &     
               workplaces_percent_change_from_baseline != "" &           
               residential_percent_change_from_baseline != ""         
               ) %>% 
    # Summarise for every county, how many complete rows of mobility data we have
    group_by(country_region_code, sub_region_1, sub_region_2) %>% 
    summarise(NROW_MOBIL = sum(IsCompleteRow), PCTCOMP_MOBIL = mean(IsCompleteRow)) %>% 
    ungroup %>% 
    # This is dumb but it's just to fix some joining/naming issues:
    mutate(
        sub_region_1_join = sub_region_1,
        sub_region_2_join = ifelse(sub_region_2 == "", "District of Columbia", sub_region_2)
        )

# Join these summary stats onto the base df
df = df %>% 
    left_join(mobil, by = c("State" = "sub_region_1_join", "County" = "sub_region_2_join"))

# Filter to population >= 100000, and pick top number of cases
pick = df %>% 
    filter(Pop >= POP_CUTOFF, NROW_MOBIL > 30) %>% 
    arrange(desc(!!rlang::sym(PICK_VAR))) %>% 
    ungroup %>% 
    slice(1:PICK_N)

View(pick)

# Select the columns that are relevant as a list
pick_out = pick %>% 
    select(FIPS, State, County, country_region_code, sub_region_1, sub_region_2 = sub_region_2, CurDate, CurCases, CurCasesRate, CurDeaths, CurDeathsRate, Pop, Density)

fwrite(pick, "Top50LargeCounties_0423.csv")
