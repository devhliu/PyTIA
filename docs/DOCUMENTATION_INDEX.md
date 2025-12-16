# PyTIA Documentation Index

## 📋 Start Here

| Document | Purpose | Audience | Time |
|----------|---------|----------|------|
| [QUICK_START.md](QUICK_START.md) | 30-second quickstart | Everyone | 2 min |
| [README.md](README.md) | Project overview | New users | 5 min |
| [LICENSE](LICENSE) | MIT License | Legal | - |

## 🚀 Usage Guides

| Document | Purpose | Level |
|----------|---------|-------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Complete usage guide (400+ lines) | Beginner → Advanced |
| [docs/CONFIG.md](docs/CONFIG.md) | Configuration reference (450+ lines) | Reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture | Intermediate → Advanced |
| [examples/README.md](examples/README.md) | Examples quick start | Beginner |

## 📚 Examples

| File | Type | Purpose |
|------|------|---------|
| [examples/example_multitime.py](examples/example_multitime.py) | Python | Multi-timepoint demo |
| [examples/example_stp.py](examples/example_stp.py) | Python | STP all 3 methods |
| [examples/demo_stp_calculations.py](examples/demo_stp_calculations.py) | Python | STP calculations |
| [examples/config_multitime.yaml](examples/config_multitime.yaml) | YAML | Multi-timepoint template |
| [examples/config_stp_phys.yaml](examples/config_stp_phys.yaml) | YAML | Physical decay template |
| [examples/config_stp_haenscheid.yaml](examples/config_stp_haenscheid.yaml) | YAML | Hänscheid template |
| [examples/config_stp_prior_seg.yaml](examples/config_stp_prior_seg.yaml) | YAML | Segmentation template |

## 🛠️ Technical Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) | STP technical details | Developers |
| [docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md) | Implementation notes | Developers |
| [PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md) | Project status | Project managers |

## 📖 Design Documents

| File | Purpose |
|------|---------|
| [docs/PyTIA-Design.md](docs/PyTIA-Design.md) | Original design |
| [docs/PyTIA-Design-Final.md](docs/PyTIA-Design-Final.md) | Final design |
| [docs/PyTIA-STP-design.md](docs/PyTIA-STP-design.md) | STP design |
| [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) | STP implementation |

## 🎯 Quick Reference by Task

### "I want to run TIA estimation"

**Via CLI:**
1. Read [QUICK_START.md](QUICK_START.md) (2 min)
2. Create config from [examples/](examples/) templates
3. Run: `pytia run --config config.yaml`

**Via Python:**
1. Read [QUICK_START.md](QUICK_START.md) (2 min)
2. Run [examples/example_multitime.py](examples/example_multitime.py)
3. Modify for your data

### "I need to configure PyTIA"

1. [QUICK_START.md](QUICK_START.md) — Minimal config (2 min)
2. Copy from [examples/config_*.yaml](examples/) (5 min)
3. Read [docs/CONFIG.md](docs/CONFIG.md) for detailed options (30 min)
4. Validate: `pytia validate --config config.yaml`

### "I want to use the Python API"

1. [QUICK_START.md](QUICK_START.md) — Basic syntax (2 min)
2. [examples/example_multitime.py](examples/example_multitime.py) — Real example (10 min)
3. [examples/example_stp.py](examples/example_stp.py) — STP examples (10 min)
4. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Section 5: Python API (15 min)

### "I need STP (single-timepoint) calculation"

1. [QUICK_START.md](QUICK_START.md) — STP section (3 min)
2. [examples/example_stp.py](examples/example_stp.py) — See all 3 methods (10 min)
3. Pick config template:
   - Physical: [examples/config_stp_phys.yaml](examples/config_stp_phys.yaml)
   - Hänscheid: [examples/config_stp_haenscheid.yaml](examples/config_stp_haenscheid.yaml)
   - Segmentation: [examples/config_stp_prior_seg.yaml](examples/config_stp_prior_seg.yaml)
4. [docs/CONFIG.md](docs/CONFIG.md) — single_time section (15 min)

### "I'm troubleshooting an issue"

1. [QUICK_START.md](QUICK_START.md) — Troubleshooting section
2. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Section 9: Troubleshooting
3. Validate config: `pytia validate --config config.yaml`
4. Show config: `pytia info --config config.yaml`
5. Check [docs/CONFIG.md](docs/CONFIG.md) for option meanings

### "I want to understand the system"

1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture (20 min)
2. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Section 1: Overview (10 min)
3. [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) — STP details (15 min)

## 📍 File Navigation Map

### Root Level

```
/
├── README.md               ← Start here (project overview)
├── QUICK_START.md          ← Start here (30-sec quickstart)
├── LICENSE                 ← MIT License
├── PROJECT_COMPLETION_SUMMARY.md  ← What was delivered
└── DOCUMENTATION_INDEX.md  ← This file
```

### docs/ Folder

```
docs/
├── USER_GUIDE.md           ← How to use (400+ lines)
├── CONFIG.md               ← Config reference (450+ lines)
├── ARCHITECTURE.md         ← System design
├── SINGLE_TIMEPOINT_IMPLEMENTATION.md
├── IMPLEMENTATION_COMPLETE.md
├── [design documents]      ← Original designs
└── [version docs]
```

### examples/ Folder

```
examples/
├── README.md               ← Examples guide
├── example_multitime.py    ← Python API demo
├── example_stp.py          ← STP methods demo
├── demo_stp_calculations.py
├── config_multitime.yaml
├── config_stp_phys.yaml
├── config_stp_haenscheid.yaml
└── config_stp_prior_seg.yaml
```

### pytia/ Folder

```
pytia/
├── __init__.py             ← Public API exports
├── cli.py                  ← CLI interface (run, validate, info)
├── engine.py               ← Core computation
├── config.py               ← Config management
├── io.py                   ← I/O utilities
├── types.py                ← Results dataclass
├── models/                 ← Model implementations
├── classify.py
├── denoise.py
├── masking.py
├── metrics.py
├── noise.py
├── uncertainty.py
└── [other modules]
```

## 🔍 Find by Topic

### Installation & Setup
- [README.md](README.md#installation) — Installation instructions
- [QUICK_START.md](QUICK_START.md#installation) — Quick setup

### CLI Usage
- [QUICK_START.md](QUICK_START.md#cli-commands) — CLI commands
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md#cli-details) — Detailed CLI guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#command-line-interface) — CLI architecture

### Python API
- [QUICK_START.md](QUICK_START.md#option-2-python) — Quick Python example
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md#python-api) — Python API guide
- [examples/example_multitime.py](examples/example_multitime.py) — Example code
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#python-api) — API architecture

### Configuration
- [QUICK_START.md](QUICK_START.md#minimal-config-yaml) — Minimal config
- [docs/CONFIG.md](docs/CONFIG.md) — Complete reference
- [examples/config_*.yaml](examples/) — Config templates
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#config-driven-architecture) — Config architecture

### Single-Timepoint (STP)
- [QUICK_START.md](QUICK_START.md#single-timepoint-stp-modes) — STP overview
- [examples/example_stp.py](examples/example_stp.py) — STP examples
- [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) — STP details
- [examples/config_stp_*.yaml](examples/) — STP configs

### Multi-Timepoint
- [examples/example_multitime.py](examples/example_multitime.py) — Multi-timepoint example
- [examples/config_multitime.yaml](examples/config_multitime.yaml) — Multi-timepoint config
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Multi-timepoint workflows

### Troubleshooting
- [QUICK_START.md](QUICK_START.md#troubleshooting) — Quick fixes
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md#troubleshooting) — Detailed troubleshooting
- [docs/CONFIG.md](docs/CONFIG.md#validation-instructions) — Config validation

### Examples
- [examples/README.md](examples/README.md) — Examples guide
- [examples/example_*.py](examples/) — Python examples
- [examples/config_*.yaml](examples/) — Config examples

## 📊 Documentation Statistics

| Category | Files | Lines |
|----------|-------|-------|
| User Guides | 3 | 900+ |
| Configuration | 4 | 250+ |
| Examples | 3 | 600+ |
| Design Docs | 5 | 1000+ |
| Technical Docs | 3 | 800+ |
| API Documentation | 1 | 100+ |
| **Total** | **19** | **3650+** |

## ⏱️ Reading Time Guide

| Time | Reading Path |
|------|--------------|
| **2 min** | QUICK_START.md |
| **5 min** | README.md |
| **15 min** | QUICK_START.md + one example |
| **30 min** | docs/USER_GUIDE.md sections 1-5 |
| **1 hour** | docs/USER_GUIDE.md + docs/CONFIG.md |
| **2 hours** | Full docs + examples + architecture |

## 🎓 Learning Paths

### Path 1: Quick User (15 min)
1. [QUICK_START.md](QUICK_START.md) (2 min)
2. Copy [examples/config_multitime.yaml](examples/config_multitime.yaml) (2 min)
3. Run: `pytia run --config config.yaml` (2 min)
4. Check output in output/ folder (5 min)
5. Modify for your data (2 min)

### Path 2: CLI Power User (30 min)
1. [QUICK_START.md](QUICK_START.md) (2 min)
2. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) section 4 (15 min)
3. [docs/CONFIG.md](docs/CONFIG.md) — skim sections (10 min)
4. Try all three commands (3 min)

### Path 3: Python Developer (1 hour)
1. [QUICK_START.md](QUICK_START.md) (2 min)
2. [examples/example_multitime.py](examples/example_multitime.py) (15 min)
3. [examples/example_stp.py](examples/example_stp.py) (15 min)
4. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) section 5 (15 min)
5. Modify examples for your data (13 min)

### Path 4: Advanced User (2 hours)
1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (30 min)
2. [docs/USER_GUIDE.md](docs/USER_GUIDE.md) (30 min)
3. [docs/CONFIG.md](docs/CONFIG.md) (30 min)
4. [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) (20 min)

## 🔗 Cross-References

### Common Questions → Documents

| Question | Document | Section |
|----------|----------|---------|
| How do I install? | README.md | Installation |
| How do I run TIA? | QUICK_START.md | Quick Start |
| What are the CLI commands? | QUICK_START.md | CLI Commands |
| How do I configure? | docs/CONFIG.md | All sections |
| What's the Python API? | docs/USER_GUIDE.md | Section 5 |
| How do STP methods work? | docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md | All |
| What do the outputs mean? | docs/USER_GUIDE.md | Section 8 |
| Something's wrong | QUICK_START.md | Troubleshooting |

## 📋 Checklist for Getting Started

- [ ] Read QUICK_START.md (2 min)
- [ ] Install PyTIA: `pip install pytia`
- [ ] Copy example config: `cp examples/config_multitime.yaml my_config.yaml`
- [ ] Edit my_config.yaml with your image paths
- [ ] Validate: `pytia validate --config my_config.yaml`
- [ ] Run: `pytia run --config my_config.yaml`
- [ ] Check output in output/ folder
- [ ] Read docs/USER_GUIDE.md for advanced options
- [ ] Explore examples/ folder
- [ ] Check docs/ARCHITECTURE.md for system understanding

## 🆘 Where to Find Help

**For quick answers:**
- [QUICK_START.md](QUICK_START.md) — Fastest answers
- [QUICK_START.md#troubleshooting](QUICK_START.md#troubleshooting) — Common issues

**For detailed explanations:**
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Complete guide
- [docs/CONFIG.md](docs/CONFIG.md) — All options explained

**For examples:**
- [examples/README.md](examples/README.md) — How to run examples
- [examples/example_*.py](examples/) — Code examples
- [examples/config_*.yaml](examples/) — Config examples

**For architecture/design:**
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design
- [docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md](docs/SINGLE_TIMEPOINT_IMPLEMENTATION.md) — STP design

**For implementation details:**
- [pytia/__init__.py](../pytia/__init__.py) — API code
- [pytia/cli.py](../pytia/cli.py) — CLI code
- [pytia/engine.py](../pytia/engine.py) — Core computation
- [tests/](../tests/) — Usage patterns

---

**Last Updated:** 2025  
**Status:** ✅ Complete  
**Total Documentation:** 3650+ lines across 19 files  
