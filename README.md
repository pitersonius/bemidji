# Voice Notes

Bemidji is an app for summarizing voice recordings. As of right now, it can convert your speech to text, attach time stamps to it and later let you download a text file with both your entire speech in plain text form and also the key moments. 
This web app is supposed to be run locally to keep all of the user analyzed conversations confidential. Please notice that because of this app's focus on being run locally, it will be up to you to choose an LLM to use with it, which will also affect the speed and the quality of end result.

Functionality:
- Support of 5 languages (English, Finnish, Russian, Swedish, Norwegian) and unofficial support of more than 100 world languages.
- Various types of LLMs, from which users can choose the one that suits their current needs best.
- Everything is happening locally, so there is no need for being connected to the internet.
- App comes with several models pre-installed, but make sure to install your model of choice in the Models menu on the top right.

How to run (Linux and MacOS):
- Download or clone the repo.
- Install the latest verstion of LM Studio and run it alongside the app.
- Move over to the repository's folder in terminal.
- Execute "./run.sh". If it doesn't work, execute "chmod +x ./run.sh"
- If you don't get a browser window pop up, open any web browser manually and enter "localhost:5050" in the search bar on top.
- All of your recordings, transcriptions and key moments are saved in their respectful folders. You can access these files at any times, and append to your existing voice notes to give your new voice notes more context.
