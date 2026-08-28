import {
  ArrowRight,
  Check,
  ChevronDown,
  Gauge,
  Home,
  Search,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://localhost:5000" : "");

const properties = [
  {
    name: "North Harbor Residence",
    detail: "4 bed • 3 bath • 3,250 sq ft",
    price: "$1.48M",
    color: "blue",
    tag: "Shortlisted",
  },
  {
    name: "Willow Terrace",
    detail: "3 bed • 2 bath • 2,180 sq ft",
    price: "$1.02M",
    color: "green",
    tag: "Low risk",
  },
];

function App() {
  const [query, setQuery] = useState(
    "What documents verify legal ownership of a property?",
  );
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploadState, setUploadState] = useState(
    "Drop a file here or choose one to index",
  );
  const [selected, setSelected] = useState([true, true]);
  const [searchMode, setSearchMode] = useState("Buy");
  const [bedrooms, setBedrooms] = useState("3+");
  const [liveProperties, setLiveProperties] = useState(properties);
  const [marketStats, setMarketStats] = useState({
    change: "+12.8% Q3",
    values: [30, 45, 59, 73, 53, 68],
    labels: ["Jan", "Mar", "May", "Jul", "Sep"],
  });
  const [comparison, setComparison] = useState([]);
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    async function loadWorkspaceData() {
      try {
        const [propertyResponse, statsResponse, documentsResponse] =
          await Promise.all([
            fetch(
              `${API_BASE}/properties?mode=${searchMode}&bedrooms=${bedrooms}`,
            ),
            fetch(`${API_BASE}/market-stats`),
            fetch(`${API_BASE}/documents`),
          ]);
        if (!propertyResponse.ok || !statsResponse.ok)
          throw new Error("Workspace data is unavailable.");
        const propertyData = await propertyResponse.json();
        const statsData = await statsResponse.json();
        const documentsData = await documentsResponse.json();
        setLiveProperties(propertyData.properties || []);
        setSelected((propertyData.properties || []).map(() => true));
        setMarketStats(statsData);
        setDocuments(documentsData.documents || []);
      } catch (requestError) {
        setError(requestError.message);
      }
    }
    loadWorkspaceData();
  }, [searchMode, bedrooms]);

  async function loadComparison() {
    const ids = liveProperties
      .filter((_, index) => selected[index])
      .map((property) => property.id);
    if (!ids.length) {
      setComparison([]);
      return;
    }
    const response = await fetch(`${API_BASE}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ property_ids: ids }),
    });
    if (response.ok) setComparison((await response.json()).properties || []);
  }

  async function askQuestion(event) {
    event?.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleanQuery }),
      });
      const data = await response.json();
      if (!response.ok || data.status === "error")
        throw new Error(data.message || "The assistant could not answer.");
      setAnswer(data);
      document
        .querySelector("#assistant")
        ?.scrollIntoView({ behavior: "smooth" });
    } catch (requestError) {
      setError(
        `${requestError.message} Check that the backend is running on ${API_BASE}.`,
      );
    } finally {
      setLoading(false);
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadState(`Indexing ${file.name}...`);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || data.status === "error")
        throw new Error(data.message || "Upload failed.");
      setUploadState(`${file.name} is ready for search`);
      const documentsResponse = await fetch(`${API_BASE}/documents`);
      if (documentsResponse.ok)
        setDocuments((await documentsResponse.json()).documents || []);
    } catch (requestError) {
      setUploadState("Upload could not be completed");
      setError(requestError.message);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="IntelliHomes home">
          <span className="brand-mark">
            <Home size={14} />
          </span>{" "}
          IntelliHomes
        </a>
        <nav>
          <a href="#search">Discover</a>
          <a href="#assistant">AI Assistant</a>
          <a href="#documents">Documents</a>
          <a href="#compare">Compare</a>
        </nav>
        <button
          className="dark-button"
          onClick={() =>
            document
              .querySelector("#assistant")
              ?.scrollIntoView({ behavior: "smooth" })
          }
        >
          Launch Demo <ArrowRight size={13} />
        </button>
      </header>

      <main id="top">
        <section className="hero content-grid" id="search">
          <div className="hero-copy reveal">
            <p className="eyebrow">AI-powered real estate assistant</p>
            <h1>
              Find the right home
              <br />
              <em>with clarity, speed,</em> and confidence.
            </h1>
            <p className="lede">
              Search properties, decode legal documents, and ground every
              decision in source-backed evidence.
            </p>
            <div className="hero-actions">
              <button
                className="primary-button"
                onClick={() =>
                  document
                    .querySelector("#compare")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                Start searching <ArrowRight size={15} />
              </button>
              <button
                className="text-button"
                onClick={() =>
                  document
                    .querySelector("#assistant")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                See the AI in action
              </button>
            </div>
            <div className="stats">
              <div>
                <strong>1200+</strong>
                <span>verified listings</span>
              </div>
              <div>
                <strong>98</strong>
                <span>% citation accuracy</span>
              </div>
              <div>
                <strong>24</strong>
                <span>minute document reviews</span>
              </div>
            </div>
          </div>
          <div className="search-preview reveal delay-one">
            <div className="preview-top">
              <span>
                <i /> Live intelligence
              </span>
              <small>3 min ago</small>
            </div>
            <form onSubmit={askQuestion}>
              <label>Search properties or ask a question</label>
              <div className="search-input">
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
                <button aria-label="Search" disabled={loading}>
                  <Search size={15} />
                </button>
              </div>
            </form>
            <div className="mini-result">
              <strong>Modern Loft</strong>
              <span>North Loop • $1.24M</span>
              <b>AI Match</b>
            </div>
            <div className="mini-result active">
              <strong>Garden Villa</strong>
              <span>Westfield • $1.71M</span>
              <b>Top pick</b>
            </div>
          </div>
        </section>

        <section className="section" id="workflow">
          <SectionTitle
            eyebrow="Everything in one workspace"
            title="From first search to final signature, the flow feels effortless."
          />
          <div className="three-up">
            <Feature
              icon={<Search size={14} />}
              title="Smart search"
              text="Refine by budget, neighborhood, school rating, and commute patterns."
            />
            <Feature
              icon={<Sparkles size={14} />}
              title="Document intelligence"
              text="Upload title deeds, sale agreements, and permits for instant summaries with citations."
            />
            <Feature
              icon={<Gauge size={14} />}
              title="Instant compare"
              text="Side-by-side scoring for investment potential, location, and legal clarity."
            />
          </div>
        </section>

        <section className="section" id="dashboard">
          <SectionTitle
            eyebrow="Dashboard experience"
            title="Designed like a premium real-estate operating system."
          />
          <div className="dashboard-grid">
            <div className="chart-panel panel">
              <div className="panel-heading">
                <strong>Market pulse</strong>
                <span className="positive">{marketStats.change}</span>
              </div>
              <div className="bars">
                {marketStats.values.map((height, index) => (
                  <span
                    key={height}
                    className={index === 3 ? "highlight" : ""}
                    style={{ height: `${height}%` }}
                  />
                ))}
              </div>
              <div className="axis">
                <span>Jan</span>
                <span>Mar</span>
                <span>May</span>
                <span>Jul</span>
                <span>Sep</span>
              </div>
            </div>
            <div className="panel saved">
              <div className="panel-heading">
                <strong>Saved properties</strong>
                <a href="#compare">View all</a>
              </div>
              <p>
                Skyline Residences <b>$1.42M</b>
              </p>
              <p>
                Harbor House <b>$980K</b>
              </p>
              <p>
                Oak Grove Villa <b>$1.18M</b>
              </p>
            </div>
            <div className="panel recent">
              <div className="panel-heading">
                <strong>Recent documents</strong>
                <b>Ready</b>
              </div>
              {(documents.length ? documents : (
                [{ filename: "No uploads yet", chunks_indexed: 0 }]
              )
              )
                .slice(0, 3)
                .map((document) => (
                  <p key={`${document.filename}-${document.created_at}`}>
                    {document.filename}{" "}
                    <span>
                      {document.chunks_indexed ?
                        `${document.chunks_indexed} chunks`
                      : "Ready"}
                    </span>
                  </p>
                ))}
            </div>
          </div>
        </section>

        <section className="section shortlist" id="compare">
          <SectionTitle
            eyebrow="Search experience"
            title="Build a shortlist with one thought-out flow."
          />
          <div className="shortlist-grid">
            <aside className="filters panel">
              <strong>Search mode</strong>
              <div className="segmented">
                {["Buy", "Rent", "Invest"].map((mode) => (
                  <button
                    key={mode}
                    className={searchMode === mode ? "selected" : ""}
                    onClick={() => setSearchMode(mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>
              <strong>Bedrooms</strong>
              <div className="segmented">
                {["2+", "3+", "4+"].map((count) => (
                  <button
                    key={count}
                    className={bedrooms === count ? "selected" : ""}
                    onClick={() => setBedrooms(count)}
                  >
                    {count}
                  </button>
                ))}
              </div>
            </aside>
            <div className="property-list">
              {liveProperties.map((property, index) => (
                <PropertyCard
                  key={property.name}
                  property={property}
                  checked={selected[index]}
                  onToggle={() =>
                    setSelected((current) =>
                      current.map((value, itemIndex) =>
                        itemIndex === index ? !value : value,
                      ),
                    )
                  }
                />
              ))}
            </div>
          </div>
        </section>

        <section className="section assistant" id="assistant">
          <SectionTitle
            eyebrow="AI assistant"
            title="Ask questions like you would ask a world-class advisor."
          />
          <div className="assistant-grid">
            <div>
              <p className="assistant-intro">
                Ask about ownership, legal safety, documents, or your next move.
                The answer below is retrieved from your indexed property
                sources.
              </p>
              <form className="question-form" onSubmit={askQuestion}>
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  aria-label="Ask your property question"
                />
                <button className="primary-button" disabled={loading}>
                  {loading ? "Thinking..." : "Ask"} <ArrowRight size={14} />
                </button>
              </form>
              {error && <p className="error-message">{error}</p>}
              {answer && <Answer answer={answer} />}
            </div>
            <div className="source-card panel">
                <div className="panel-heading">
                  <strong>Source-backed answers</strong>
                  <Check size={15} />
                </div>
                <ul>
                  <li>Document citations with page references</li>
                  <li>Markdown-ready summaries</li>
                  <li>Conversation history and follow-up prompts</li>
                </ul>
                <small>Compliance with real property laws</small>
            </div>
          </div>
        </section>

        <section className="section" id="documents">
          <SectionTitle
            eyebrow="Document workflow"
            title="Upload, summarize, and review legal paperwork in moments."
          />
          <div className="document-grid">
            <label className="upload-panel panel">
              <input
                type="file"
                accept=".pdf,.md,.txt,.html"
                onChange={uploadDocument}
              />
              <Upload size={20} />
              <strong>Upload documents</strong>
              <span>{uploadState}</span>
              <small>PDF, Markdown, text, and HTML files are supported.</small>
            </label>
            <div className="panel summary-panel">
              <div className="panel-heading">
                <strong>Live summary</strong>
                <span className="positive">Citations ready</span>
              </div>
              <p>
                “The agreement includes a 15-day inspection window, a fixed
                transfer fee, and one unresolved clause related to parking
                rights.”
              </p>
              <span className="summary-note">4 pages highlighted</span>
            </div>
          </div>
        </section>

        <section className="section compare-section">
          <SectionTitle
            eyebrow="Property comparison"
            title="Compare homes with transparent scoring across every signal."
          />
          <div className="compare-grid">
            {(comparison.length ? comparison : (
              liveProperties.filter((_, index) => selected[index])
            )
            ).map((property) => (
              <div className="panel compare-card" key={property.name}>
                <strong>{property.name}</strong>
                <span>Investment score {property.score}</span>
                <ul>
                  <li>Location score {property.location_score || "Pending"}</li>
                  <li>Legal score {property.legal_score || "Pending"}</li>
                  <li>
                    Overall recommendation:{" "}
                    {property.recommendation || "Select compare"}
                  </li>
                </ul>
              </div>
            ))}
            {selected.every((value) => !value) && (
              <div className="panel compare-card empty-compare">
                Select a property above to compare it here.
              </div>
            )}
          </div>
          <button
            className="primary-button compare-action"
            onClick={loadComparison}
          >
            Refresh comparison scores <Gauge size={13} />
          </button>
        </section>
      </main>
      <footer>
        <span>
          IntelliHomes
          <br />
          <small>Luxury-level real-estate workflows made simple.</small>
        </span>
        <span>
          Back to top
          <br />
          <small>© 2026 IntelliHomes</small>
        </span>
      </footer>
    </div>
  );
}

function SectionTitle({ eyebrow, title }) {
  return (
    <div className="section-title">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
    </div>
  );
}
function Feature({ icon, title, text }) {
  return (
    <article className="feature-card panel">
      <span className="feature-icon">{icon}</span>
      <strong>{title}</strong>
      <p>{text}</p>
    </article>
  );
}
function PropertyCard({ property, checked, onToggle }) {
  return (
    <article className="property-card panel">
      <div className={`property-image ${property.color}`}>
        <Home size={25} />
      </div>
      <div className="property-info">
        <strong>{property.name}</strong>
        <span>{property.detail}</span>
        <div>
          <b>{property.tag}</b>
          <small>Park nearby</small>
        </div>
        <strong>{property.price}</strong>
      </div>
      <button className="compare-button" onClick={onToggle}>
        {checked ?
          <Check size={13} />
        : <X size={13} />}{" "}
        Compare
      </button>
    </article>
  );
}
function Answer({ answer }) {
  return (
    <div className="answer-panel panel">
      <div className="answer-heading">
        <span className="status-dot" /> Grounded answer{" "}
        <small>{answer.sources?.length || 0} sources</small>
      </div>
      <p>{answer.answer || "No grounded answer returned."}</p>
      {answer.sources?.length > 0 && (
        <details>
          <summary>
            Inspect retrieved evidence <ChevronDown size={14} />
          </summary>
          <div className="citations">
            {answer.sources.slice(0, 3).map((source, index) => (
              <div className="citation" key={source.id || index}>
                <strong>
                  [{index + 1}] {source.source || source.id}
                </strong>
                <span>
                  {source.section || "Source chunk"} · relevance{" "}
                  {Math.round((source.score || 0) * 100)}%
                </span>
                <p>{source.text}</p>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
