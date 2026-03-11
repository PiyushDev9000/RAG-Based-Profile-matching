from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
import random
import os

# ──────────────────────────────────────────────
# Initialize Faker
# ──────────────────────────────────────────────
fake = Faker()
random.seed(42)  # For reproducibility

# ──────────────────────────────────────────────
# Domain Profiles
# Each domain has: skills, job titles, education options, companies
# ──────────────────────────────────────────────
DOMAIN_PROFILES = {
    "Data Scientist": {
        "skills": [
            "Python", "Machine Learning", "TensorFlow", "PyTorch", "SQL",
            "Pandas", "NumPy", "Scikit-learn", "Data Visualization", "Statistics",
            "Deep Learning", "NLP", "Computer Vision", "Spark", "Tableau"
        ],
        "titles": [
            "Junior Data Scientist", "Data Scientist", "Senior Data Scientist",
            "ML Engineer", "AI Researcher", "Data Science Lead"
        ],
        "education": [
            "B.S. Computer Science", "M.S. Data Science",
            "M.S. Computer Science", "Ph.D. Machine Learning",
            "B.S. Statistics", "M.S. Artificial Intelligence"
        ],
        "companies": [
            "Google", "Meta", "Netflix", "Uber", "Airbnb", "OpenAI",
            "DeepMind", "Apple", "Amazon", "DataRobot"
        ]
    },
    "Backend Engineer": {
        "skills": [
            "Python", "Java", "Node.js", "REST APIs", "PostgreSQL",
            "MongoDB", "Docker", "Kubernetes", "AWS", "Microservices",
            "Redis", "GraphQL", "Spring Boot", "FastAPI", "MySQL"
        ],
        "titles": [
            "Junior Backend Developer", "Backend Engineer",
            "Senior Backend Engineer", "Software Engineer II",
            "API Developer", "Platform Engineer"
        ],
        "education": [
            "B.S. Computer Science", "B.S. Software Engineering",
            "M.S. Computer Science", "B.S. Information Technology",
            "B.S. Computer Engineering"
        ],
        "companies": [
            "Amazon", "Microsoft", "Spotify", "Twitter", "LinkedIn",
            "Stripe", "Shopify", "Twilio", "GitHub", "Cloudflare"
        ]
    },
    "Marketing Manager": {
        "skills": [
            "SEO", "Content Marketing", "Google Analytics", "Social Media",
            "Email Marketing", "Campaign Management", "Copywriting",
            "Market Research", "HubSpot", "Brand Strategy",
            "Google Ads", "Facebook Ads", "Influencer Marketing",
            "CRM", "A/B Testing"
        ],
        "titles": [
            "Marketing Coordinator", "Marketing Manager",
            "Senior Marketing Manager", "Digital Marketing Specialist",
            "Growth Marketing Manager", "Content Strategist"
        ],
        "education": [
            "B.S. Marketing", "B.A. Communications",
            "MBA Marketing", "B.S. Business Administration",
            "B.A. Journalism", "M.S. Digital Marketing"
        ],
        "companies": [
            "Nike", "Coca-Cola", "HubSpot", "Salesforce", "P&G",
            "Unilever", "Adobe", "Mailchimp", "Hootsuite", "Ogilvy"
        ]
    },
    "DevOps Engineer": {
        "skills": [
            "Docker", "Kubernetes", "AWS", "CI/CD", "Jenkins",
            "Terraform", "Linux", "Python", "Ansible", "Monitoring",
            "Prometheus", "Grafana", "Git", "Bash Scripting", "Azure"
        ],
        "titles": [
            "DevOps Engineer", "Site Reliability Engineer",
            "Cloud Engineer", "Infrastructure Engineer",
            "Senior DevOps Engineer", "Platform Engineer"
        ],
        "education": [
            "B.S. Computer Science", "B.S. Information Technology",
            "M.S. Cloud Computing", "B.S. Computer Engineering",
            "B.S. Network Engineering"
        ],
        "companies": [
            "AWS", "Google Cloud", "IBM", "Red Hat", "HashiCorp",
            "Cloudflare", "Datadog", "PagerDuty", "Atlassian", "GitLab"
        ]
    },
    "Finance Analyst": {
        "skills": [
            "Financial Modeling", "Excel", "SQL", "Python", "Bloomberg Terminal",
            "Forecasting", "Risk Analysis", "Valuation", "PowerBI", "Accounting",
            "DCF Analysis", "Budgeting", "SAP", "Tableau", "Financial Reporting"
        ],
        "titles": [
            "Financial Analyst", "Junior Analyst", "Senior Financial Analyst",
            "Investment Analyst", "FP&A Analyst", "Corporate Finance Analyst"
        ],
        "education": [
            "B.S. Finance", "B.S. Economics", "MBA Finance",
            "M.S. Financial Engineering", "CFA", "B.S. Accounting",
            "M.S. Economics"
        ],
        "companies": [
            "Goldman Sachs", "JP Morgan", "BlackRock", "Morgan Stanley",
            "Deloitte", "McKinsey", "Bain", "PwC", "Citadel", "Fidelity"
        ]
    },
    "Product Manager": {
        "skills": [
            "Product Roadmap", "Agile", "Scrum", "User Research", "Jira",
            "A/B Testing", "Data Analysis", "Stakeholder Management",
            "Wireframing", "SQL", "Figma", "OKRs", "Go-to-Market Strategy",
            "Competitive Analysis", "PRD Writing"
        ],
        "titles": [
            "Associate Product Manager", "Product Manager",
            "Senior Product Manager", "Group Product Manager",
            "Director of Product", "Principal PM"
        ],
        "education": [
            "B.S. Computer Science", "MBA", "B.S. Business",
            "M.S. Human Computer Interaction", "B.S. Engineering",
            "M.S. Product Design"
        ],
        "companies": [
            "Apple", "Google", "Facebook", "Atlassian", "Figma",
            "Notion", "Slack", "Zoom", "Dropbox", "Asana"
        ]
    }
}


# ──────────────────────────────────────────────
# STEP 1: Generate Resume Content (as a dictionary)
# ──────────────────────────────────────────────
def generate_resume_content(domain, profile):
    """
    Builds all resume content for a given domain profile.
    Returns a dictionary with name, skills, experiences, education etc.
    
    - Uses Faker for realistic personal details
    - Randomly samples skills so each resume is unique
    - Generates 2-3 jobs with chronological year ranges
    """

    # Personal details
    name = fake.name()
    email = fake.email()
    phone = fake.phone_number()
    city = fake.city()

    # Randomly pick 7-9 skills from the domain's skill list
    num_skills = random.randint(7, min(9, len(profile["skills"])))
    skills = random.sample(profile["skills"], k=num_skills)

    # Generate 2-3 work experiences (most recent first)
    experiences = []
    current_year = 2024
    num_jobs = random.randint(2, 3)

    for i in range(num_jobs):
        duration = random.randint(1, 3)
        start_year = current_year - duration
        end_year = "Present" if i == 0 else str(current_year)
        current_year = start_year  # Move back in time for next job

        experiences.append({
            "title": random.choice(profile["titles"]),
            "company": random.choice(profile["companies"]),
            "start": start_year,
            "end": end_year,
            "bullets": [
                fake.sentence(nb_words=random.randint(10, 14)),
                fake.sentence(nb_words=random.randint(9, 13)),
                fake.sentence(nb_words=random.randint(10, 12))
            ]
        })

    # Education
    education = random.choice(profile["education"])
    grad_year = random.randint(2012, 2020)
    university = fake.company() + " University"

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "city": city,
        "domain": domain,
        "skills": skills,
        "experiences": experiences,
        "education": education,
        "university": university,
        "grad_year": grad_year
    }


# ──────────────────────────────────────────────
# STEP 2: Write Resume Content to PDF
# ──────────────────────────────────────────────
def create_resume_pdf(content, output_path):
    """
    Takes a resume content dictionary and writes it to a PDF file.
    
    - Uses bold section headers (SKILLS, EXPERIENCE, EDUCATION)
      so the chunker in Checkpoint 2 can detect section boundaries
    - Uses reportlab's Platypus layout engine (story-based PDF building)
    """

    # Set up the PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )

    # ── Define Text Styles ──────────────────────
    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        'NameStyle',
        parent=styles['Normal'],
        fontSize=18,
        fontName='Helvetica-Bold',
        spaceAfter=4,
        textColor=HexColor('#1A1A2E')
    )
    contact_style = ParagraphStyle(
        'ContactStyle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=HexColor('#555555'),
        spaceAfter=6
    )
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        spaceBefore=14,
        spaceAfter=5,
        textColor=HexColor('#2C3E50'),
        borderPad=2
    )
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        spaceAfter=3,
        leading=14
    )
    job_title_style = ParagraphStyle(
        'JobTitle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        spaceAfter=2
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        leftIndent=15,
        spaceAfter=2,
        leading=13
    )

    # ── Build the PDF Story (list of elements) ──
    story = []

    # Name
    story.append(Paragraph(content["name"], name_style))

    # Contact Info
    story.append(Paragraph(
        f'{content["email"]}  |  {content["phone"]}  |  {content["city"]}',
        contact_style
    ))
    story.append(Spacer(1, 0.1 * inch))

    # ── SKILLS Section ──────────────────────────
    # NOTE: "SKILLS" header is bold & uppercase
    # Checkpoint 2 chunker will split on these headers
    story.append(Paragraph("SKILLS", section_header_style))
    story.append(Paragraph(", ".join(content["skills"]), normal_style))

    # ── EXPERIENCE Section ──────────────────────
    story.append(Paragraph("EXPERIENCE", section_header_style))
    for exp in content["experiences"]:
        # Job title line
        story.append(Paragraph(
            f'{exp["title"]}  |  {exp["company"]}  |  {exp["start"]} – {exp["end"]}',
            job_title_style
        ))
        # Bullet points
        for bullet in exp["bullets"]:
            story.append(Paragraph(f"• {bullet}", bullet_style))
        story.append(Spacer(1, 0.05 * inch))

    # ── EDUCATION Section ───────────────────────
    story.append(Paragraph("EDUCATION", section_header_style))
    story.append(Paragraph(
        f'{content["education"]}  |  {content["university"]}  |  {content["grad_year"]}',
        normal_style
    ))

    # Build & save the PDF
    doc.build(story)


# ──────────────────────────────────────────────
# STEP 3: Main Loop — Generate All Resumes
# ──────────────────────────────────────────────
def generate_all_resumes(output_dir="resumes", resumes_per_domain=6):
    """
    Loops through all domain profiles and generates resumes.
    6 domains x 6 resumes = 36 total resumes (satisfies 30+ requirement)
    
    Saves each resume as: resumes/firstname_lastname.pdf
    """

    os.makedirs(output_dir, exist_ok=True)
    total = 0
    generated_names = set()  # Avoid duplicate filenames

    for domain, profile in DOMAIN_PROFILES.items():
        print(f"\n📁 Generating {resumes_per_domain} resumes for: {domain}")

        count = 0
        attempts = 0

        while count < resumes_per_domain and attempts < resumes_per_domain * 3:
            attempts += 1
            content = generate_resume_content(domain, profile)

            # Ensure unique filename
            base_name = content["name"].lower().replace(" ", "_").replace(".", "")
            if base_name in generated_names:
                continue
            generated_names.add(base_name)

            filename = base_name + ".pdf"
            output_path = os.path.join(output_dir, filename)

            try:
                create_resume_pdf(content, output_path)
                print(f"  ✅ {filename}  [{domain}]")
                count += 1
                total += 1
            except Exception as e:
                print(f"  ❌ Failed to create {filename}: {e}")

    print(f"\n🎉 Done! Generated {total} resumes → /{output_dir}/")
    print(f"📊 Domains covered: {len(DOMAIN_PROFILES)}")
    print(f"📄 Resumes per domain: {resumes_per_domain}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_resumes()