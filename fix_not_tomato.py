from pathlib import Path

path = Path("templates/index.html")
html = path.read_text(encoding="utf-8")

marker = "function buildResultCard(d) {"

if marker not in html:
    raise SystemExit("ERROR: buildResultCard() was not found. No changes made.")

if "NOT_TOMATO_UPLOAD_MESSAGE" in html:
    print("The Not Tomato handling is already installed.")
else:
    replacement = """function buildResultCard(d) {

      // NOT_TOMATO_UPLOAD_MESSAGE
      // Stop diagnosis when the uploaded image is not a tomato leaf.
      if (
        d.predicted_class === "Not_Tomato" ||
        d.disease === "Not Tomato Leaf" ||
        d.disease === "Not_Tomato"
      ) {
        return `
          <div style="
            max-width:760px;
            margin:30px auto;
            padding:42px 28px;
            text-align:center;
            border:1px solid rgba(239,68,68,.3);
            border-radius:24px;
            background:linear-gradient(135deg,rgba(255,245,242,.96),rgba(255,255,255,.88));
            box-shadow:0 15px 45px rgba(40,60,80,.10);
          ">
            <div style="font-size:54px;margin-bottom:15px;">🍅</div>
            <h2 style="
              color:#d94841;
              font-size:clamp(23px,4vw,32px);
              font-weight:800;
              margin:0 0 14px;
            ">Oops! This is not a tomato leaf.</h2>
            <p style="
              color:#536579;
              font-size:16px;
              line-height:1.8;
              max-width:520px;
              margin:0 auto 22px;
            ">
              Please upload a clear image of a tomato leaf to continue
              with disease detection.
            </p>
            <div style="
              display:inline-block;
              padding:11px 19px;
              border-radius:30px;
              background:#fff0eb;
              color:#c94b3d;
              font-size:13px;
              font-weight:700;
            ">Image verification failed</div>
          </div>
        `;
      }
"""

    html = html.replace(marker, replacement, 1)
    path.write_text(html, encoding="utf-8")
    print("SUCCESS: Non-tomato warning installed.")

