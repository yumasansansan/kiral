// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of KIRAL. The SPDX lines name its copyright holder and its
// license, in the machine-readable form of the REUSE specification: the GNU
// General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// KIRAL's addon in a Fcitx 5 of the test's own -- its test frontend, its test
// UI and the addon, nothing else, and no user's files -- typed into as an
// application's input context is. With the dictionary alone (`testkiral
// dictionary`) what is shown and committed is known: kana is 仮名, which Enter
// commits; Space and Space show the list with かな selected, a click on カナ
// commits it; Escape deletes what is typed; a key held with Control passes;
// and what is shown is committed as another input method takes the context.
// With a model (`testkiral model`), the session's wake-up, called on the
// engine's worker thread, has Fcitx's thread show the refinement.

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fcitx-utils/event.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/eventloopinterface.h>
#include <fcitx-utils/handlertable.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/testing.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextmanager.h>
#include <fcitx/inputmethodgroup.h>
#include <fcitx/inputmethodmanager.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/text.h>
#include <testfrontend_public.h>

namespace {

// What the input context shows where the text is typed: the preedit the
// application draws, or Fcitx's.
std::string preeditOf(fcitx::InputContext &inputContext) {
    const std::string client = inputContext.inputPanel().clientPreedit().toString();
    return client.empty() ? inputContext.inputPanel().preedit().toString() : client;
}

// Types keys, each a key's name as Fcitx reads it, and checks that the input
// method took each, or did not.
void type(fcitx::AddonInstance &frontend, fcitx::ICUUID uuid, const std::vector<std::string> &keys, bool taken = true) {
    for (const auto &key : keys) {
        const bool accepted = frontend.call<fcitx::ITestFrontend::sendKeyEvent>(uuid, fcitx::Key(key), false);
        if (accepted != taken) {
            FCITX_FATAL() << key << (taken ? ": not taken" : ": taken");
        }
    }
}

void check(bool truth, std::string_view what) {
    if (!truth) {
        FCITX_FATAL() << what;
    }
}

// An input context of the test frontend's, KIRAL its input method and the test
// input method beside it.
fcitx::ICUUID inputContext(fcitx::Instance &instance, fcitx::AddonInstance &frontend) {
    auto group = instance.inputMethodManager().currentGroup();
    group.inputMethodList().clear();
    group.inputMethodList().emplace_back("kiral");
    group.inputMethodList().emplace_back("testim");
    group.setDefaultInputMethod("kiral");
    instance.inputMethodManager().setGroup(group);
    return frontend.call<fcitx::ITestFrontend::createInputContext>("testapp");
}

void testDictionary(fcitx::Instance &instance) {
    auto &frontend = *instance.addonManager().addon("testfrontend");
    const auto uuid = inputContext(instance, frontend);
    auto &context = *instance.inputContextManager().findByUUID(uuid);
    std::vector<std::string> commits;
    const auto watcher =
        instance.watchEvent(fcitx::EventType::InputContextCommitString, fcitx::EventWatcherPhase::Default,
                            [&commits](fcitx::Event &event) {
                                commits.push_back(static_cast<fcitx::CommitStringEvent &>(event).text());
                            });
    // kana: 仮名 shown as it is typed, and committed by Enter.
    type(frontend, uuid, {"k", "a", "n"});
    check(preeditOf(context) == "可ｎ", "kan shows 可ｎ, its か converted as it is typed");
    type(frontend, uuid, {"a"});
    check(preeditOf(context) == "仮名", "kana shows 仮名");
    frontend.call<fcitx::ITestFrontend::pushCommitExpectation>("仮名");
    type(frontend, uuid, {"Return"});
    check(preeditOf(context).empty(), "nothing shown once committed");
    // Space converts, Space again shows the list and selects かな.
    type(frontend, uuid, {"k", "a", "n", "a", "space"});
    check(!context.inputPanel().candidateList(), "no list before the second Space");
    type(frontend, uuid, {"space"});
    const auto list = context.inputPanel().candidateList();
    check(list && list->size() == 3, "the list of 仮名, かな and カナ");
    check(list->candidate(0).text().toString() == "仮名" && list->candidate(1).text().toString() == "かな" &&
              list->candidate(2).text().toString() == "カナ",
          "仮名, かな and カナ in order");
    check(list->cursorIndex() == 1 && preeditOf(context) == "かな", "かな selected and shown");
    frontend.call<fcitx::ITestFrontend::pushCommitExpectation>("かな");
    type(frontend, uuid, {"Return"});
    // A candidate clicked is committed.
    type(frontend, uuid, {"k", "a", "n", "a", "space", "space"});
    frontend.call<fcitx::ITestFrontend::pushCommitExpectation>("カナ");
    context.inputPanel().candidateList()->candidate(2).select(&context);
    check(preeditOf(context).empty() && !context.inputPanel().candidateList(), "nothing shown once clicked");
    // Escape deletes what is typed, committing nothing.
    type(frontend, uuid, {"k", "a", "n", "Escape"});
    check(preeditOf(context).empty(), "nothing shown after Escape");
    // Keys with nothing typed but characters, and keys held with Control, are
    // the application's.
    type(frontend, uuid, {"space", "Return", "BackSpace", "Control+a"}, false);
    // What is shown is committed as another input method takes the context.
    type(frontend, uuid, {"k", "a", "n", "a"});
    frontend.call<fcitx::ITestFrontend::pushCommitExpectation>("仮名");
    instance.setCurrentInputMethod(&context, "testim", true);
    check(preeditOf(context).empty(), "nothing shown once another input method has the context");
    check(commits == std::vector<std::string>{"仮名", "かな", "カナ", "仮名"}, "仮名, かな, カナ and 仮名 committed");
    frontend.call<fcitx::ITestFrontend::destroyInputContext>(uuid);
}

// What the test with a model keeps while Fcitx's loop runs.
struct Model {
    int updates = 0;
    int typed = 0;
    std::uint64_t deadline = 0;
    std::unique_ptr<fcitx::HandlerTableEntry<fcitx::EventHandler>> watcher;
    std::unique_ptr<fcitx::EventSourceTime> waiting;
};

// The time between two looks at what is shown, and how long the test looks, in
// microseconds.
constexpr std::uint64_t pace = 10'000;
constexpr std::uint64_t patience = 5'000'000;

void testModel(fcitx::Instance &instance, Model &model) {
    auto &frontend = *instance.addonManager().addon("testfrontend");
    const auto uuid = inputContext(instance, frontend);
    // Each update of the input panel, a key's and a refinement's.
    model.watcher = instance.watchEvent(fcitx::EventType::InputContextUpdateUI, fcitx::EventWatcherPhase::Default,
                                        [&model](fcitx::Event &) { ++model.updates; });
    type(frontend, uuid, {"k", "a", "n", "a"});
    model.typed = model.updates;
    check(model.typed > 0, "the keys update the input panel");
    // Fcitx's loop runs until the wake-up's poll has shown a refinement: the
    // input panel updated once more than the keys did.
    model.deadline = fcitx::now(CLOCK_MONOTONIC) + patience;
    model.waiting = instance.eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, fcitx::now(CLOCK_MONOTONIC) + pace, pace / 10,
        [&instance, &frontend, &model, uuid](fcitx::EventSourceTime *source, std::uint64_t now) {
            if (model.updates == model.typed && now < model.deadline) {
                source->setNextInterval(pace);
                source->setOneShot();
                return true;
            }
            check(model.updates > model.typed, "a refinement shown within five seconds");
            type(frontend, uuid, {"space", "Return"});
            check(preeditOf(*instance.inputContextManager().findByUUID(uuid)).empty(), "nothing shown once committed");
            frontend.call<fcitx::ITestFrontend::destroyInputContext>(uuid);
            instance.exit();
            return true;
        });
}

} // namespace

int main(int argc, char **argv) {
    // The test's name, from C's arguments.
#pragma clang unsafe_buffer_usage begin
    const std::string which = argc == 2 ? argv[1] : "";
#pragma clang unsafe_buffer_usage end
    if (which != "dictionary" && which != "model") {
        FCITX_FATAL() << "usage: testkiral dictionary|model";
    }
    fcitx::setupTestingEnvironmentPath(KIRAL_TEST_BINARY_DIR, {KIRAL_TEST_BINARY_DIR, KIRAL_FCITX_ADDON_DIR},
                                       {std::filesystem::path("test") / which, KIRAL_FCITX_TESTING_DIR});
    char program[] = "testkiral";
    char disable[] = "--disable=all";
    char enable[] = "--enable=testfrontend,testui,testim,kiral";
    char *arguments[] = {program, disable, enable};
    fcitx::Instance instance(3, arguments);
    instance.addonManager().registerDefaultLoader(nullptr);
    Model model;
    instance.eventDispatcher().schedule([&instance, &model, which] {
        if (which == "dictionary") {
            testDictionary(instance);
            instance.exit();
        } else {
            testModel(instance, model);
        }
    });
    return instance.exec();
}
