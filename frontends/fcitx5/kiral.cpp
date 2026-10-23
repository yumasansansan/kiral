// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// KIRAL's addon for Fcitx 5: an input method engine that translates between
// Fcitx 5 and KIRAL's C ABI (crates/kiral-capi/include/kiral.h), and does
// nothing else. The engine opens kiral/dictionary.kdic and, where there is
// one, kiral/model.kmdl from Fcitx's data directories ($XDG_DATA_HOME/fcitx5,
// then the system's); each input context has a session of it. A key goes to
// the session, which says whether it took it; what the session commits, shows
// and offers goes to Fcitx. The session's wake-up, called on the engine's
// worker thread once a refinement is given back, schedules the input
// context's poll on Fcitx's thread, for an input context and an engine that
// are both still there.

#include <kiral.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <fcitx-utils/capabilityflags.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/keysym.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/standardpaths.h>
#include <fcitx-utils/textformatflags.h>
#include <fcitx-utils/trackableobject.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextmanager.h>
#include <fcitx/inputcontextproperty.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/inputmethodentry.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/text.h>
#include <fcitx/userinterface.h>

namespace kiral {
namespace {

// KIRAL's code of a key Fcitx gives: one of the keys that type no character,
// or the character a key types; 0 for a key that is neither.
std::uint32_t codeOf(const fcitx::Key &key) {
    const auto sym = static_cast<std::uint32_t>(key.sym());
    switch (sym) {
    case FcitxKey_space:
    case FcitxKey_KP_Space:
        return KIRAL_KEY_SPACE;
    case FcitxKey_Return:
    case FcitxKey_KP_Enter:
        return KIRAL_KEY_ENTER;
    case FcitxKey_BackSpace:
        return KIRAL_KEY_BACKSPACE;
    case FcitxKey_Delete:
    case FcitxKey_KP_Delete:
        return KIRAL_KEY_DELETE;
    case FcitxKey_Escape:
        return KIRAL_KEY_ESCAPE;
    case FcitxKey_Tab:
        return KIRAL_KEY_TAB;
    case FcitxKey_Left:
    case FcitxKey_KP_Left:
        return KIRAL_KEY_LEFT;
    case FcitxKey_Right:
    case FcitxKey_KP_Right:
        return KIRAL_KEY_RIGHT;
    case FcitxKey_Up:
    case FcitxKey_KP_Up:
        return KIRAL_KEY_UP;
    case FcitxKey_Down:
    case FcitxKey_KP_Down:
        return KIRAL_KEY_DOWN;
    case FcitxKey_Home:
    case FcitxKey_KP_Home:
        return KIRAL_KEY_HOME;
    case FcitxKey_End:
    case FcitxKey_KP_End:
        return KIRAL_KEY_END;
    case FcitxKey_Page_Up:
    case FcitxKey_KP_Page_Up:
        return KIRAL_KEY_PAGE_UP;
    case FcitxKey_Page_Down:
    case FcitxKey_KP_Page_Down:
        return KIRAL_KEY_PAGE_DOWN;
    default:
        break;
    }
    if (sym >= FcitxKey_F1 && sym <= FcitxKey_F24) {
        return KIRAL_KEY_F1 + (sym - FcitxKey_F1);
    }
    return fcitx::Key::keySymToUnicode(key.sym());
}

// KIRAL's modifiers of the states Fcitx gives: Super for every logo key's.
std::uint32_t modifiersOf(fcitx::KeyStates states) {
    std::uint32_t modifiers = 0;
    if (states.test(fcitx::KeyState::Shift)) {
        modifiers |= KIRAL_SHIFT;
    }
    if (states.test(fcitx::KeyState::Ctrl)) {
        modifiers |= KIRAL_CONTROL;
    }
    if (states.test(fcitx::KeyState::Alt)) {
        modifiers |= KIRAL_ALT;
    }
    if (states.testAny(fcitx::KeyStates{fcitx::KeyState::Super, fcitx::KeyState::Super2, fcitx::KeyState::Hyper,
                                        fcitx::KeyState::Hyper2, fcitx::KeyState::Meta})) {
        modifiers |= KIRAL_SUPER;
    }
    return modifiers;
}

// A string the C ABI returned, which lives until the session changes, as a
// string of its own.
std::string stringOf(const char *text, std::size_t length) {
    return length == 0 ? std::string() : std::string(std::string_view(text, length));
}

class Engine;

// An input context's session of the engine.
class State final : public fcitx::InputContextProperty {
public:
    State(Engine &engine, fcitx::InputContext &inputContext);
    ~State() override;
    State(const State &) = delete;
    State &operator=(const State &) = delete;
    State(State &&) = delete;
    State &operator=(State &&) = delete;

    // The session; NULL where the engine did not open, which the C ABI takes
    // as nothing done.
    kiral_session *session() const { return session_; }

    // On the worker's thread: has Fcitx's thread take the refinement.
    void wake();

private:
    fcitx::TrackableObjectReference<Engine> engine_;
    fcitx::TrackableObjectReference<fcitx::InputContext> inputContext_;
    fcitx::EventDispatcher &dispatcher_;
    kiral_session *session_;
};

// The input method engine: KIRAL's engine, and its sessions as properties of
// the input contexts.
class Engine final : public fcitx::InputMethodEngineV2, public fcitx::TrackableObject<Engine> {
public:
    explicit Engine(fcitx::Instance *instance [[clang::lifetimebound]]);
    ~Engine() override;
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    Engine(Engine &&) = delete;
    Engine &operator=(Engine &&) = delete;

    void keyEvent(const fcitx::InputMethodEntry &entry, fcitx::KeyEvent &keyEvent) override;
    void deactivate(const fcitx::InputMethodEntry &entry, fcitx::InputContextEvent &event) override;
    void reset(const fcitx::InputMethodEntry &entry, fcitx::InputContextEvent &event) override;

    fcitx::Instance &instance() const { return *instance_; }
    kiral_engine *engine() const { return engine_; }

    // Takes an input context's refinements that have come, and shows them.
    void poll(fcitx::InputContext &inputContext);
    // Commits a candidate clicked in the list.
    void choose(fcitx::InputContext &inputContext, std::size_t index);

private:
    // Commits what the session committed, and shows what it shows.
    void update(fcitx::InputContext &inputContext, State &state, bool committed);
    // The list of candidates the session has the frontend show, if any.
    std::unique_ptr<fcitx::CandidateList> candidates(kiral_session *session);
    // Gives the session the text before the cursor, where the application
    // gives it, as the left context of what is typed next.
    static void giveContext(fcitx::InputContext &inputContext, State &state);

    fcitx::Instance *instance_;
    kiral_engine *engine_ = nullptr;
    fcitx::FactoryFor<State> factory_;
};

// A candidate in Fcitx's list: clicked, the session commits it.
class Candidate final : public fcitx::CandidateWord {
public:
    Candidate(Engine &engine [[clang::lifetimebound]], std::size_t index, std::string text);
    ~Candidate() override;
    Candidate(const Candidate &) = delete;
    Candidate &operator=(const Candidate &) = delete;
    Candidate(Candidate &&) = delete;
    Candidate &operator=(Candidate &&) = delete;

    void select(fcitx::InputContext *inputContext) const override;

private:
    Engine &engine_;
    std::size_t index_;
};

// The wake-up the C ABI calls on the engine's worker thread, with the state
// the session is of; the C ABI does not call it once the session is freed.
extern "C" {
static void wakeUp(void *state) { static_cast<State *>(state)->wake(); }
}

State::State(Engine &engine, fcitx::InputContext &inputContext)
    : engine_(engine.watch()), inputContext_(inputContext.watch()),
      dispatcher_(engine.instance().eventDispatcher()), session_(kiral_session_new(engine.engine())) {
    kiral_session_set_notify(session_, wakeUp, this);
}

State::~State() { kiral_session_free(session_); }

void State::wake() {
    // Fcitx's dispatcher takes functions from any thread; the function runs
    // on Fcitx's thread, while the engine and the input context are there.
    dispatcher_.scheduleWithContext(engine_, [engine = engine_, inputContext = inputContext_] {
        if (auto *context = inputContext.get()) {
            engine.get()->poll(*context);
        }
    });
}

Candidate::Candidate(Engine &engine, std::size_t index, std::string text)
    : fcitx::CandidateWord(fcitx::Text(std::move(text))), engine_(engine), index_(index) {}

Candidate::~Candidate() = default;

void Candidate::select(fcitx::InputContext *inputContext) const { engine_.choose(*inputContext, index_); }

Engine::Engine(fcitx::Instance *instance)
    : instance_(instance),
      factory_([this](fcitx::InputContext &inputContext) { return new State(*this, inputContext); }) {
    const auto &paths = fcitx::StandardPaths::global();
    const std::filesystem::path dictionary = paths.locate(fcitx::StandardPathsType::PkgData, "kiral/dictionary.kdic");
    const std::filesystem::path model = paths.locate(fcitx::StandardPathsType::PkgData, "kiral/model.kmdl");
    if (dictionary.empty()) {
        FCITX_ERROR() << "KIRAL: no kiral/dictionary.kdic in Fcitx's data directories";
    } else {
        std::array<char, 512> error{};
        engine_ = kiral_engine_open(dictionary.c_str(), model.empty() ? nullptr : model.c_str(), error.data(),
                                    error.size());
        if (engine_ == nullptr) {
            FCITX_ERROR() << "KIRAL: " << error.data();
        } else if (model.empty()) {
            FCITX_INFO() << "KIRAL: " << dictionary.string() << ", no kiral/model.kmdl: the dictionary's drafts alone";
        } else {
            FCITX_INFO() << "KIRAL: " << dictionary.string() << " and " << model.string();
        }
    }
    instance_->inputContextManager().registerProperty("kiralState", &factory_);
}

Engine::~Engine() {
    // The sessions first: once they are freed no wake-up is called, and the
    // engine closes with no session of it left.
    factory_.unregister();
    kiral_engine_close(engine_);
}

void Engine::keyEvent(const fcitx::InputMethodEntry &entry, fcitx::KeyEvent &keyEvent) {
    FCITX_UNUSED(entry);
    if (engine_ == nullptr || keyEvent.isRelease()) {
        return;
    }
    auto &inputContext = *keyEvent.inputContext();
    if (inputContext.capabilityFlags().testAny(fcitx::CapabilityFlag::PasswordOrSensitive)) {
        return;
    }
    const std::uint32_t code = codeOf(keyEvent.key());
    if (code == 0) {
        return;
    }
    auto &state = *inputContext.propertyFor(&factory_);
    std::size_t typed = 0;
    kiral_session_preedit(state.session(), &typed);
    if (typed == 0) {
        giveContext(inputContext, state);
    }
    if (!kiral_session_key(state.session(), code, modifiersOf(keyEvent.key().states()))) {
        return;
    }
    keyEvent.filterAndAccept();
    update(inputContext, state, true);
}

void Engine::deactivate(const fcitx::InputMethodEntry &entry, fcitx::InputContextEvent &event) {
    // What is shown is committed rather than lost, as another input method
    // takes the input context, or another window the focus.
    auto &inputContext = *event.inputContext();
    auto &state = *inputContext.propertyFor(&factory_);
    std::size_t length = 0;
    const char *shown = kiral_session_preedit(state.session(), &length);
    if (length > 0) {
        inputContext.commitString(stringOf(shown, length));
    }
    reset(entry, event);
}

void Engine::reset(const fcitx::InputMethodEntry &entry, fcitx::InputContextEvent &event) {
    FCITX_UNUSED(entry);
    auto &inputContext = *event.inputContext();
    kiral_session_reset(inputContext.propertyFor(&factory_)->session());
    inputContext.inputPanel().reset();
    inputContext.updatePreedit();
    inputContext.updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void Engine::poll(fcitx::InputContext &inputContext) {
    auto &state = *inputContext.propertyFor(&factory_);
    if (kiral_session_poll(state.session())) {
        update(inputContext, state, false);
    }
}

void Engine::choose(fcitx::InputContext &inputContext, std::size_t index) {
    auto &state = *inputContext.propertyFor(&factory_);
    if (kiral_session_choose(state.session(), index)) {
        update(inputContext, state, true);
    }
}

void Engine::update(fcitx::InputContext &inputContext, State &state, bool committed) {
    kiral_session *session = state.session();
    std::size_t length = 0;
    if (committed) {
        // The text a key or a click committed: the session keeps it until the
        // next, so that a poll after it commits nothing.
        const char *commit = kiral_session_commit(session, &length);
        if (length > 0) {
            inputContext.commitString(stringOf(commit, length));
        }
    }
    const std::string text = stringOf(kiral_session_preedit(session, &length), length);
    fcitx::Text preedit;
    std::size_t start = 0;
    std::size_t end = 0;
    const bool highlighted = kiral_session_preedit_highlight(session, &start, &end);
    if (!highlighted) {
        start = text.size();
        end = text.size();
    }
    const fcitx::TextFormatFlags underlined{fcitx::TextFormatFlag::Underline};
    if (start > 0) {
        preedit.append(text.substr(0, start), underlined);
    }
    if (end > start) {
        preedit.append(text.substr(start, end - start),
                       fcitx::TextFormatFlags{fcitx::TextFormatFlag::Underline, fcitx::TextFormatFlag::HighLight});
    }
    if (text.size() > end) {
        preedit.append(text.substr(end), underlined);
    }
    preedit.setCursor(static_cast<int>(kiral_session_preedit_cursor(session)));
    auto &panel = inputContext.inputPanel();
    if (inputContext.capabilityFlags().test(fcitx::CapabilityFlag::Preedit)) {
        panel.setClientPreedit(preedit);
        panel.setPreedit(fcitx::Text());
    } else {
        panel.setPreedit(preedit);
        panel.setClientPreedit(fcitx::Text());
    }
    panel.setCandidateList(candidates(session));
    inputContext.updatePreedit();
    inputContext.updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

std::unique_ptr<fcitx::CandidateList> Engine::candidates(kiral_session *session) {
    const std::size_t count = kiral_session_candidates(session);
    if (count == 0 || !kiral_session_candidates_shown(session)) {
        return nullptr;
    }
    auto list = std::make_unique<fcitx::CommonCandidateList>();
    list->setPageSize(9);
    list->setLayoutHint(fcitx::CandidateLayoutHint::Vertical);
    for (std::size_t index = 0; index < count; ++index) {
        std::size_t length = 0;
        const char *text = kiral_session_candidate(session, index, &length);
        list->append(std::make_unique<Candidate>(*this, index, stringOf(text, length)));
    }
    list->setGlobalCursorIndex(static_cast<int>(kiral_session_candidate_selected(session)));
    return list;
}

void Engine::giveContext(fcitx::InputContext &inputContext, State &state) {
    if (!inputContext.capabilityFlags().test(fcitx::CapabilityFlag::SurroundingText)) {
        return;
    }
    const auto &surrounding = inputContext.surroundingText();
    if (!surrounding.isValid()) {
        return;
    }
    const std::string &text = surrounding.text();
    const std::size_t characters = fcitx::utf8::length(text);
    if (characters == fcitx::utf8::INVALID_LENGTH || surrounding.cursor() > characters) {
        return;
    }
    const auto before = fcitx::utf8::ncharByteLength(text.begin(), surrounding.cursor());
    kiral_session_set_context(state.session(), text.data(), static_cast<std::size_t>(before));
}

// Makes the engine for Fcitx's addon manager.
class Factory final : public fcitx::AddonFactory {
public:
    Factory() = default;
    ~Factory() override;
    Factory(const Factory &) = delete;
    Factory &operator=(const Factory &) = delete;
    Factory(Factory &&) = delete;
    Factory &operator=(Factory &&) = delete;

    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override;
};

Factory::~Factory() = default;

fcitx::AddonInstance *Factory::create(fcitx::AddonManager *manager) { return new Engine(manager->instance()); }

} // namespace
} // namespace kiral

// Fcitx's macro defines the function its addon manager looks up, which no
// header declares, and returns its factory, a static object.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
FCITX_ADDON_FACTORY_V2(kiral, kiral::Factory)
#pragma clang diagnostic pop
