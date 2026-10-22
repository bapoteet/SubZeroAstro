import subprocess, re, json, sys
pdf=sys.argv[1]
n=int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo",pdf],capture_output=True,text=True).stdout).group(1))
titles={1:"Welcome to SubZeroAstro",2:"What you need",3:"Installing SubZeroAstro",4:"First-time setup",5:"Finding your way around",6:"Page 1",7:"Page 2",8:"Page 3",9:"Page 4",10:"Recipes for common",11:"Troubleshooting and",12:"Keyboard shortcuts",13:"Files and folders",14:"Privacy, licence",15:"Processing Techniques",16:"Index"}
pages={}
for pg in range(2,n+1):
    t=subprocess.run(["pdftotext","-f",str(pg),"-l",str(pg),"-layout",pdf,"-"],capture_output=True,text=True).stdout
    for k,v in titles.items():
        if re.search(rf"^{k}\s+{re.escape(v)}", t, re.M):
            pages[str(k)]=pg          # last occurrence wins (the contents pages come first)
print(n, pages); json.dump(pages,open("pages.json","w"))
